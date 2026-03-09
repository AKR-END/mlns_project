"""
Extended probing experiments across ESM-2 model sizes (8M, 35M, 650M):
1. Ridge (linear) probe on last-layer embeddings
2. MLP probes (1 and 2 hidden layers) on last-layer embeddings
3. Learned layer weighting with linear head
4. Learned layer weighting with MLP head

Builds on feasibility_test.py — reuses data loading, embeddings, and feature extraction.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.environ["TORCH_HOME"] = "/ssd_scratch/akr/model_cache"
sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import (
    download_davis, load_davis, compute_ligand_fingerprints,
    extract_esm2_embeddings, build_features, evaluate_probe,
    DEVICE, EMBED_DIR,
)


# ── MLP Probe ─────────────────────────────────────────────────────────────────
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, hidden_dims, lr=1e-3,
              epochs=100, batch_size=512, patience=10, dropout=0.1):
    input_dim = X_train.shape[1]
    model = MLPProbe(input_dim, hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE).eval()
    print(f"    Trained for {epoch+1} epochs (best at epoch {epoch+1-wait})")
    return model


def evaluate_mlp(model, X_test, y_test, name="MLP"):
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()

    r, p_val = stats.pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  {name:45s} | r={r:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    return {"name": name, "pearson_r": r, "p_value": p_val, "rmse": rmse, "r2": r2}


# ── Learned Layer Weighting ───────────────────────────────────────────────────
class LayerWeightedProbe(nn.Module):
    """Learns a scalar mixture over all ESM-2 layers, then applies a linear or MLP head."""
    def __init__(self, num_layers, embed_dim, ligand_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

        input_dim = embed_dim + ligand_dim
        if hidden_dims is None or len(hidden_dims) == 0:
            self.head = nn.Linear(input_dim, 1)
        else:
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.head = nn.Sequential(*layers)

    def forward(self, layer_embeddings, ligand_fp):
        weights = torch.softmax(self.layer_weights, dim=0)
        protein_emb = self.layer_scale * torch.einsum("bld,l->bd", layer_embeddings, weights)
        combined = torch.cat([protein_emb, ligand_fp], dim=1)
        return self.head(combined).squeeze(-1)


def build_all_layer_features(df, embed_dict, fp_dict):
    """Build feature tensors with all layer embeddings stacked."""
    sample_prot = next(iter(embed_dict.values()))
    num_layers = max(sample_prot.keys()) + 1
    embed_dim = sample_prot[0].shape[0]

    all_layer_embs = []
    ligand_fps = []
    targets = []

    for _, row in df.iterrows():
        pname = row["protein_name"]
        lname = row["ligand_name"]

        layer_stack = torch.stack([embed_dict[pname][l] for l in range(num_layers)])
        all_layer_embs.append(layer_stack)
        ligand_fps.append(torch.FloatTensor(fp_dict[lname]))
        targets.append(row["pKd"])

    return (torch.stack(all_layer_embs), torch.stack(ligand_fps),
            np.array(targets), num_layers, embed_dim)


def train_layer_weighted(layer_embs_train, fp_train, y_train,
                         layer_embs_val, fp_val, y_val,
                         num_layers, embed_dim, ligand_dim,
                         hidden_dims=None, lr=1e-3, epochs=100,
                         batch_size=512, patience=10):
    model = LayerWeightedProbe(num_layers, embed_dim, ligand_dim, hidden_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(layer_embs_train, fp_train, torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for emb_b, fp_b, y_b in train_loader:
            emb_b, fp_b, y_b = emb_b.to(DEVICE), fp_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(emb_b, fp_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(layer_embs_val.to(DEVICE), fp_val.to(DEVICE))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(DEVICE)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE).eval()
    print(f"    Trained for {epoch+1} epochs (best at epoch {epoch+1-wait})")
    return model


def evaluate_layer_weighted(model, layer_embs_test, fp_test, y_test, name="LayerWeighted"):
    model.eval()
    with torch.no_grad():
        y_pred = model(layer_embs_test.to(DEVICE), fp_test.to(DEVICE)).cpu().numpy()

    r, p_val = stats.pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  {name:45s} | r={r:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    weights = torch.softmax(model.layer_weights.data.cpu(), dim=0).numpy()
    top3 = np.argsort(weights)[-3:][::-1]
    weight_str = ", ".join(f"L{i}={weights[i]:.3f}" for i in top3)
    print(f"    Top-3 layer weights: {weight_str}")

    return {"name": name, "pearson_r": r, "p_value": p_val, "rmse": rmse, "r2": r2,
            "layer_weights": weights}


# ── 650M embedding extraction (manual, handles missing regression head) ───────
def extract_650m_all_layers(df):
    """Extract all-layer embeddings for 650M model (handles offline regression head)."""
    embed_path = os.path.join(EMBED_DIR, "esm2_t33_650M_UR50D_all_layers_davis.pt")
    if os.path.exists(embed_path):
        print(f"Loading cached 650M all-layer embeddings from {embed_path}")
        return torch.load(embed_path, map_location="cpu", weights_only=True)

    import esm

    model_ckpt = "/ssd_scratch/akr/model_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt"
    print(f"Loading 650M model from {model_ckpt}...")
    model_data = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        "esm2_t33_650M_UR50D", model_data, None)
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    unique_prots = df[["protein_name", "sequence"]].drop_duplicates()
    embed_dict = {}

    print(f"Extracting all-layer embeddings for {len(unique_prots)} proteins (650M)...")
    for idx, (_, row) in enumerate(unique_prots.iterrows()):
        name = row["protein_name"]
        seq = row["sequence"][:1022]

        data = [(name, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(num_layers + 1)))

        seq_len = len(seq)
        layer_embeds = {}
        for layer_idx in range(num_layers + 1):
            rep = results["representations"][layer_idx]
            residue_rep = rep[0, 1:seq_len + 1, :]
            mean_rep = residue_rep.mean(dim=0).cpu()
            layer_embeds[layer_idx] = mean_rep

        embed_dict[name] = layer_embeds

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique_prots)}] {name} (len={seq_len})")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    torch.save(embed_dict, embed_path)
    print(f"Saved all-layer 650M embeddings to {embed_path}")
    return embed_dict


# ── Run all experiments for one model ─────────────────────────────────────────
def run_experiments_for_model(model_label, embed_dict, df, fp_dict,
                              train_idx, val_idx, test_idx):
    print(f"\n{'#' * 70}")
    print(f"# {model_label}")
    print(f"{'#' * 70}")

    # -- Last-layer features --
    X_esm, _, _, y = build_features(df, embed_dict, fp_dict)
    X_train, X_val, X_test = X_esm[train_idx], X_esm[val_idx], X_esm[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    results = []

    # Ridge (linear)
    print(f"\n--- Ridge (linear) ---")
    r = evaluate_probe(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]),
                       X_test, y_test, f"{model_label} Ridge")
    results.append(r)

    # MLP probes
    print(f"\n--- MLP probes ---")
    for hidden in ([256], [512], [512, 256]):
        tag = str(hidden).replace(" ", "")
        m = train_mlp(X_train, y_train, X_val, y_val, hidden_dims=hidden)
        r = evaluate_mlp(m, X_test, y_test, f"{model_label} MLP {tag}")
        results.append(r)

    # -- Layer-weighted probes --
    layer_embs, ligand_fps, _, num_layers, embed_dim = build_all_layer_features(
        df, embed_dict, fp_dict)
    ligand_dim = ligand_fps.shape[1]
    le_train = layer_embs[train_idx]
    le_val = layer_embs[val_idx]
    le_test = layer_embs[test_idx]
    fp_tr = ligand_fps[train_idx]
    fp_vl = ligand_fps[val_idx]
    fp_te = ligand_fps[test_idx]

    print(f"\n--- Learned layer weighting (linear head) ---")
    m = train_layer_weighted(le_train, fp_tr, y_train, le_val, fp_vl, y_val,
                             num_layers, embed_dim, ligand_dim, hidden_dims=None)
    r = evaluate_layer_weighted(m, le_test, fp_te, y_test,
                                f"{model_label} LayerW+Linear")
    results.append(r)

    print(f"\n--- Learned layer weighting (MLP head) ---")
    m = train_layer_weighted(le_train, fp_tr, y_train, le_val, fp_vl, y_val,
                             num_layers, embed_dim, ligand_dim, hidden_dims=[512, 256])
    r = evaluate_layer_weighted(m, le_test, fp_te, y_test,
                                f"{model_label} LayerW+MLP[512,256]")
    results.append(r)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    download_davis()
    df = load_davis()
    fp_dict = compute_ligand_fingerprints(df)

    # 3-way split: train/val/test (64/16/20)
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    all_results = {}

    # ── ESM-2 8M ──
    embed_8m = extract_esm2_embeddings(df, model_name="esm2_t6_8M_UR50D")
    all_results["8M"] = run_experiments_for_model(
        "8M", embed_8m, df, fp_dict, train_idx, val_idx, test_idx)
    del embed_8m

    # ── ESM-2 35M ──
    embed_35m = extract_esm2_embeddings(df, model_name="esm2_t12_35M_UR50D")
    all_results["35M"] = run_experiments_for_model(
        "35M", embed_35m, df, fp_dict, train_idx, val_idx, test_idx)
    del embed_35m

    # ── ESM-2 650M ──
    embed_650m = extract_650m_all_layers(df)
    all_results["650M"] = run_experiments_for_model(
        "650M", embed_650m, df, fp_dict, train_idx, val_idx, test_idx)
    del embed_650m

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Probe':<45s} | {'r':>6s} | {'RMSE':>6s} | {'R²':>6s}")
    print("-" * 70)
    for size in ["8M", "35M", "650M"]:
        for r in all_results[size]:
            name = r["name"]
            print(f"  {name:<43s} | {r['pearson_r']:>6.4f} | {r['rmse']:>6.4f} | {r['r2']:>6.4f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
