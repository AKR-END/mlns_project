"""
4 attention methods × 3 settings for binding affinity prediction.

Attention methods:
  1. Protein-only:      attn = Linear(residue, 1)  — ligand-independent
  2. Cross-attention:    attn = softmax(lig_query @ prot_keys^T)  — ligand-dependent
  3. Self-attn + pool:   self_attn(residues) → attn = Linear(ctx, 1)
  4. Self-attn + cross:  self_attn(residues) → attn = softmax(lig @ ctx_keys^T)

Settings:
  A. MLP + Attention:          [attn_pool(residues) || ligand] → MLP
  B. MLP + Layer Weighting:    [layer_weighted(layers) || ligand] → MLP  (1 variant)
  C. MLP + LayerW + Attention: [layer_weighted || attn_pool || ligand] → MLP

Total: 4 + 1 + 4 = 9 experiments on Davis (ESM-2 650M + Morgan FP)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/ssd_scratch/akr"
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
os.environ["TORCH_HOME"] = CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import (
    download_davis, load_davis, compute_ligand_fingerprints, concordance_index,
)
from ablation_study import (
    ESM_CONFIGS, extract_residue_embeddings, extract_all_layer_embeddings,
    residue_collate_fn,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 4 Attention Modules (pluggable)
# ═══════════════════════════════════════════════════════════════════════════════

class ProteinOnlyAttn(nn.Module):
    """attn = Linear(residue_emb, 1). Ligand-independent."""
    def __init__(self, embed_dim, ligand_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, residues, ligand, mask):
        scores = self.attn(residues).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), residues).squeeze(1), weights


class CrossAttn(nn.Module):
    """Ligand queries protein: attn = softmax(lig_q @ prot_k^T / sqrt(d))."""
    def __init__(self, embed_dim, ligand_dim, attn_dim=256):
        super().__init__()
        self.q_proj = nn.Linear(ligand_dim, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.scale = attn_dim ** 0.5

    def forward(self, residues, ligand, mask):
        q = self.q_proj(ligand).unsqueeze(1)              # (B, 1, d_a)
        k = self.k_proj(residues)                          # (B, L, d_a)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / self.scale  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), residues).squeeze(1), weights


class SelfAttnPool(nn.Module):
    """Self-attention contextualizes residues, then learned pooling."""
    def __init__(self, embed_dim, ligand_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.Linear(embed_dim, 1)

    def forward(self, residues, ligand, mask):
        kpm = ~mask if mask is not None else None
        ctx, _ = self.self_attn(residues, residues, residues,
                                key_padding_mask=kpm)
        ctx = self.norm(residues + ctx)
        scores = self.pool(ctx).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), ctx).squeeze(1), weights


class SelfCrossAttn(nn.Module):
    """Self-attention → cross-attention with ligand."""
    def __init__(self, embed_dim, ligand_dim, attn_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(ligand_dim, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.scale = attn_dim ** 0.5

    def forward(self, residues, ligand, mask):
        kpm = ~mask if mask is not None else None
        ctx, _ = self.self_attn(residues, residues, residues,
                                key_padding_mask=kpm)
        ctx = self.norm(residues + ctx)
        q = self.q_proj(ligand).unsqueeze(1)
        k = self.k_proj(ctx)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), ctx).squeeze(1), weights


ATTN_METHODS = {
    "prot-only": ProteinOnlyAttn,
    "cross":     CrossAttn,
    "self+pool": SelfAttnPool,
    "self+cross": SelfCrossAttn,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Setting A: MLP + Attention
# ═══════════════════════════════════════════════════════════════════════════════

class AttnMLP(nn.Module):
    """[attn_pool(residues) || ligand] → MLP"""
    def __init__(self, attn_cls, embed_dim, ligand_dim,
                 hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.attn_module = attn_cls(embed_dim, ligand_dim)
        layers = []
        prev = embed_dim + ligand_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, residues, ligand, mask=None):
        pooled, self._weights = self.attn_module(residues, ligand, mask)
        return self.head(torch.cat([pooled, ligand], dim=1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Setting B: MLP + Layer Weighting (no attention — single variant)
# ═══════════════════════════════════════════════════════════════════════════════

class LayerWMLP(nn.Module):
    """[layer_weighted(layers) || ligand] → MLP"""
    def __init__(self, num_layers, embed_dim, ligand_dim,
                 hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        layers = []
        prev = embed_dim + ligand_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, layer_embs, ligand):
        w = torch.softmax(self.layer_weights, dim=0)
        prot = self.layer_scale * torch.einsum("bld,l->bd", layer_embs, w)
        return self.head(torch.cat([prot, ligand], dim=1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Setting C: MLP + Layer Weighting + Attention
# ═══════════════════════════════════════════════════════════════════════════════

class LayerWAttnMLP(nn.Module):
    """[layer_weighted(layers) || attn_pool(residues) || ligand] → MLP"""
    def __init__(self, attn_cls, num_layers, embed_dim, ligand_dim,
                 hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.attn_module = attn_cls(embed_dim, ligand_dim)
        layers = []
        prev = embed_dim * 2 + ligand_dim  # global + local + ligand
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, layer_embs, residues, ligand, mask=None):
        w = torch.softmax(self.layer_weights, dim=0)
        prot_global = self.layer_scale * torch.einsum("bld,l->bd", layer_embs, w)
        prot_local, self._weights = self.attn_module(residues, ligand, mask)
        return self.head(
            torch.cat([prot_global, prot_local, ligand], dim=1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════

class ResidueDS(Dataset):
    """For Setting A: per-residue + ligand."""
    def __init__(self, df, indices, res_dict, lig_dict):
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.rd, self.ld = res_dict, lig_dict
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        return (self.rd[row["protein_name"]],
                torch.FloatTensor(self.ld[row["ligand_name"]]),
                torch.tensor(row["pKd"], dtype=torch.float32))


class LayerDS(Dataset):
    """For Setting B: all-layer mean-pooled + ligand."""
    def __init__(self, df, indices, layer_dict, lig_dict, num_layers):
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.lyd, self.ld, self.nl = layer_dict, lig_dict, num_layers
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        layers = torch.stack([self.lyd[row["protein_name"]][l]
                              for l in range(self.nl)])
        return (layers, torch.FloatTensor(self.ld[row["ligand_name"]]),
                torch.tensor(row["pKd"], dtype=torch.float32))


class LayerResidueDS(Dataset):
    """For Setting C: all-layer + per-residue + ligand."""
    def __init__(self, df, indices, layer_dict, res_dict, lig_dict, num_layers):
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.lyd, self.rd, self.ld = layer_dict, res_dict, lig_dict
        self.nl = num_layers
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        layers = torch.stack([self.lyd[row["protein_name"]][l]
                              for l in range(self.nl)])
        return (layers, self.rd[row["protein_name"]],
                torch.FloatTensor(self.ld[row["ligand_name"]]),
                torch.tensor(row["pKd"], dtype=torch.float32))


def layer_residue_collate(batch):
    layers, residues, ligands, targets = zip(*batch)
    lengths = [r.shape[0] for r in residues]
    max_len = max(lengths)
    d = residues[0].shape[1]
    padded = torch.zeros(len(batch), max_len, d)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (r, l) in enumerate(zip(residues, lengths)):
        padded[i, :l] = r
        mask[i, :l] = True
    return (torch.stack(layers), padded, torch.stack(ligands),
            mask, torch.stack(targets))


# ═══════════════════════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════════════════════

def _train(model, train_loader, val_loader, forward_fn, lr=1e-3,
           epochs=100, patience=10):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred, target = forward_fn(model, batch)
            criterion(pred, target).backward()
            optimizer.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred, target = forward_fn(model, batch)
                vl += criterion(pred, target).item() * len(target)
                vn += len(target)
        val_loss = vl / vn

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    print(f"    Trained {epoch+1} ep (best {epoch+1-wait}), val={best_loss:.4f}")
    return model.to(DEVICE).eval()


# Forward functions for each setting
def fwd_A(model, batch):
    r, l, y = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
    # residue_collate gives (r, l, mask, y)
    if len(batch) == 4:
        r, l, m, y = batch
        r, l, m, y = r.to(DEVICE), l.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
        return model(r, l, m), y
    return model(r, l), y


def fwd_A_collated(model, batch):
    r, l, m, y = batch
    return model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE)), y.to(DEVICE)


def fwd_B(model, batch):
    layers, lig, y = batch
    return model(layers.to(DEVICE), lig.to(DEVICE)), y.to(DEVICE)


def fwd_C(model, batch):
    layers, res, lig, mask, y = batch
    return model(layers.to(DEVICE), res.to(DEVICE),
                 lig.to(DEVICE), mask.to(DEVICE)), y.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("ATTENTION METHODS × SETTINGS: 4 × 3 Comparison")
    print("=" * 80)

    download_davis()
    df = load_davis()
    morgan = compute_ligand_fingerprints(df)
    residue = extract_residue_embeddings(df, "650M")
    layer = extract_all_layer_embeddings(df, "650M")

    indices = np.arange(len(df))
    trv, te = train_test_split(indices, test_size=0.2, random_state=42)
    tr, va = train_test_split(trv, test_size=0.2, random_state=42)
    y_test = df.iloc[te]["pKd"].values

    cfg = ESM_CONFIGS["650M"]
    embed_dim = cfg["dim"]
    ligand_dim = 2048
    num_layers = cfg["layers"] + 1
    BS = 512
    all_results = []

    # ═══ Setting A: MLP + Attention (4 variants) ═════════════════════════════
    print("\n" + "=" * 80)
    print("SETTING A: MLP + Attention")
    print("=" * 80)

    tr_ds = ResidueDS(df, tr, residue, morgan)
    va_ds = ResidueDS(df, va, residue, morgan)
    te_ds = ResidueDS(df, te, residue, morgan)
    tr_ld = DataLoader(tr_ds, BS, shuffle=True, collate_fn=residue_collate_fn)
    va_ld = DataLoader(va_ds, BS, collate_fn=residue_collate_fn)
    te_ld = DataLoader(te_ds, BS, collate_fn=residue_collate_fn)

    for aname, acls in ATTN_METHODS.items():
        name = f"A: MLP+{aname}"
        print(f"\n  {name}")
        model = _train(
            AttnMLP(acls, embed_dim, ligand_dim),
            tr_ld, va_ld, fwd_A_collated)

        preds = []
        with torch.no_grad():
            for r, l, m, _ in te_ld:
                preds.append(model(r.to(DEVICE), l.to(DEVICE),
                                   m.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy()
        r, _ = stats.pearsonr(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        ci = concordance_index(y_test, y_pred)
        all_results.append({"name": name, "setting": "A", "attn": aname,
                            "ci": ci, "mse": mse, "r": r})
        print(f"    CI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")
        del model; torch.cuda.empty_cache()

    # ═══ Setting B: MLP + Layer Weighting (1 variant) ════════════════════════
    print("\n" + "=" * 80)
    print("SETTING B: MLP + Layer Weighting")
    print("=" * 80)

    tr_lds = LayerDS(df, tr, layer, morgan, num_layers)
    va_lds = LayerDS(df, va, layer, morgan, num_layers)
    te_lds = LayerDS(df, te, layer, morgan, num_layers)
    tr_ld_b = DataLoader(tr_lds, BS, shuffle=True)
    va_ld_b = DataLoader(va_lds, BS)
    te_ld_b = DataLoader(te_lds, BS)

    name = "B: MLP+LayerW"
    print(f"\n  {name}")
    model = _train(
        LayerWMLP(num_layers, embed_dim, ligand_dim),
        tr_ld_b, va_ld_b, fwd_B)

    preds = []
    with torch.no_grad():
        for ly, lg, _ in te_ld_b:
            preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy()
    r, _ = stats.pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    ci = concordance_index(y_test, y_pred)
    all_results.append({"name": name, "setting": "B", "attn": "none",
                        "ci": ci, "mse": mse, "r": r})
    print(f"    CI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")

    lw = torch.softmax(model.layer_weights.data.cpu(), dim=0).numpy()
    top3 = np.argsort(lw)[-3:][::-1]
    print(f"    Top layers: {', '.join(f'L{i}={lw[i]:.3f}' for i in top3)}")
    del model; torch.cuda.empty_cache()

    # ═══ Setting C: MLP + LayerW + Attention (4 variants) ════════════════════
    print("\n" + "=" * 80)
    print("SETTING C: MLP + Layer Weighting + Attention")
    print("=" * 80)

    tr_lrs = LayerResidueDS(df, tr, layer, residue, morgan, num_layers)
    va_lrs = LayerResidueDS(df, va, layer, residue, morgan, num_layers)
    te_lrs = LayerResidueDS(df, te, layer, residue, morgan, num_layers)
    tr_ld_c = DataLoader(tr_lrs, BS, shuffle=True, collate_fn=layer_residue_collate)
    va_ld_c = DataLoader(va_lrs, BS, collate_fn=layer_residue_collate)
    te_ld_c = DataLoader(te_lrs, BS, collate_fn=layer_residue_collate)

    for aname, acls in ATTN_METHODS.items():
        name = f"C: MLP+LayerW+{aname}"
        print(f"\n  {name}")
        model = _train(
            LayerWAttnMLP(acls, num_layers, embed_dim, ligand_dim),
            tr_ld_c, va_ld_c, fwd_C)

        preds = []
        with torch.no_grad():
            for ly, rs, lg, mk, _ in te_ld_c:
                preds.append(model(ly.to(DEVICE), rs.to(DEVICE),
                                   lg.to(DEVICE), mk.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy()
        r, _ = stats.pearsonr(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        ci = concordance_index(y_test, y_pred)
        all_results.append({"name": name, "setting": "C", "attn": aname,
                            "ci": ci, "mse": mse, "r": r})
        print(f"    CI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")

        lw = torch.softmax(model.layer_weights.data.cpu(), dim=0).numpy()
        top3 = np.argsort(lw)[-3:][::-1]
        print(f"    Top layers: {', '.join(f'L{i}={lw[i]:.3f}' for i in top3)}")
        del model; torch.cuda.empty_cache()

    # ═══ Summary ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FULL RESULTS")
    print("=" * 80)
    print(f"\n{'Model':<35s} | {'CI':>6s} | {'MSE':>6s} | {'r':>6s}")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['name']:<33s} | {r['ci']:>6.4f} | {r['mse']:>6.4f} | "
              f"{r['r']:>6.4f}")

    best = max(all_results, key=lambda x: x["ci"])
    print(f"\nBest: {best['name']} (CI={best['ci']:.4f})")

    # ═══ Visualization ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped by setting
    ax = axes[0]
    settings = {"A": [], "B": [], "C": []}
    for r in all_results:
        settings[r["setting"]].append(r)
    colors_s = {"A": "#42A5F5", "B": "#66BB6A", "C": "#AB47BC"}
    x_pos = 0
    xticks, xlabels = [], []
    for s in ["A", "B", "C"]:
        for r in settings[s]:
            ax.bar(x_pos, r["ci"], color=colors_s[s], alpha=0.85, edgecolor="white")
            ax.text(x_pos, r["ci"] + 0.001, f"{r['ci']:.3f}", ha="center",
                    fontsize=7, rotation=90)
            xticks.append(x_pos)
            xlabels.append(r["attn"] if r["attn"] != "none" else "LayerW")
            x_pos += 1
        x_pos += 0.5  # gap between settings
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("CI")
    ax.set_title("CI by Setting × Attention Method", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors_s[s], label=f"Setting {s}")
                       for s in "ABC"], fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Attention method comparison across settings
    ax = axes[1]
    attn_names = list(ATTN_METHODS.keys())
    for i, an in enumerate(attn_names):
        a_ci = [r["ci"] for r in all_results
                if r["attn"] == an and r["setting"] == "A"]
        c_ci = [r["ci"] for r in all_results
                if r["attn"] == an and r["setting"] == "C"]
        a_val = a_ci[0] if a_ci else 0
        c_val = c_ci[0] if c_ci else 0
        ax.plot(["Setting A", "Setting C"], [a_val, c_val],
                marker="o", linewidth=2, markersize=8, label=an)
    b_ci = [r["ci"] for r in all_results if r["setting"] == "B"][0]
    ax.axhline(y=b_ci, color="gray", linestyle="--", alpha=0.7,
               label=f"Setting B (LayerW only): {b_ci:.4f}")
    ax.set_ylabel("CI")
    ax.set_title("How LayerW Changes Each Attention Method", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("attention_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved attention_comparison.png")


if __name__ == "__main__":
    main()
