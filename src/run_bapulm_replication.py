"""
BAPULM-style replication on our stack.

Architecture exactly matches BAPULM (model/model.py from their repo):
  prot_emb (1280) ─Linear(512)─ReLU ─┐
  mol_emb  (768)  ─Linear(512)─ReLU ─┼─cat──BatchNorm──Dropout(0.1)
                                      └─→ Linear(1024,768)─ReLU
                                          ─Linear(768,512)─ReLU─Dropout
                                          ─Linear(512,32)─ReLU
                                          ─Linear(32,1)

Key differences vs. BAPULM (deliberate, document in paper):
  - We use ESM-2 650M mean-pooled (1280-d) instead of ProtT5-XL-U50 (1024-d),
    because we already have ESM-2 cached. The first Linear takes 1280→512.
  - Otherwise the head, loss, optimizer, scheduler, batch size, LR, and epoch
    budget match BAPULM exactly:
        Adam lr=1e-3, batch=256, MSE, ReduceLROnPlateau(factor=0.2, patience=5),
        90/10 random split of jglaser first 100k, 60 epochs.
  - We also apply BAPULM's preprocessing scaling on outputs (mean=6.513,
    scale=1.561) so the comparison is apples-to-apples.

Evaluation:
  Reports CI/r/RMSE/R² on the full Test2016_290 (290 complexes) AND on the
  leakage-filtered clean subset (no protein-sequence overlap with the
  jglaser-100k training set). Writes both to JSON.

Outputs:
  bapulm_replication_results.json
  bapulm_replication_predictions.json
  bapulm_replication_scatter.png
  Checkpoint: /ssd_scratch/akr/checkpoints/bapulm_replication.pt
"""

import os, sys, json, glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RESULTS_DIR, FIGURES_DIR, DATA_EXTRA_DIR,
    DEVICE, ESM_CONFIGS, CKPT_DIR,
    load_jglaser, load_pdbbind_val_test,
    extract_esm2_embeddings, compute_molformer,
    concordance_index, evaluate,
)


# ── BAPULM head architecture, with prot input dim adjustable ──
class BAPULMHead(nn.Module):
    def __init__(self, prot_dim=1280, mol_dim=768):
        super().__init__()
        self.prot_linear = nn.Linear(prot_dim, 512)
        self.mol_linear = nn.Linear(mol_dim, 512)
        self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(768, 512)
        self.linear3 = nn.Linear(512, 32)
        self.final_linear = nn.Linear(32, 1)

    def forward(self, prot, mol):
        po = torch.relu(self.prot_linear(prot))
        mo = torch.relu(self.mol_linear(mol))
        x = torch.cat([po, mo], dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        return self.final_linear(x).squeeze(-1)


class PairDS(Dataset):
    """[prot_emb, mol_emb] → standardized affinity."""
    def __init__(self, prot_names, lig_names, targets, prot_dict, mol_dict):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.pd, self.md = prot_dict, mol_dict
    def __len__(self): return len(self.t)
    def __getitem__(self, idx):
        p = self.pd[self.pn[idx]]
        m = self.md[self.ln[idx]]
        if isinstance(p, np.ndarray): p = torch.FloatTensor(p)
        if isinstance(m, np.ndarray): m = torch.FloatTensor(m)
        return p, m, torch.tensor(self.t[idx], dtype=torch.float32)


def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rs, rmses, r2s, cis = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9: continue
        rs.append(stats.pearsonr(yt, yp)[0])
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        r2s.append(r2_score(yt, yp))
        cis.append(concordance_index(yt, yp))
    def ci(arr): return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return {"r":    {"point": float(stats.pearsonr(y_true, y_pred)[0]), "ci95": ci(rs)},
            "rmse": {"point": float(np.sqrt(mean_squared_error(y_true, y_pred))), "ci95": ci(rmses)},
            "r2":   {"point": float(r2_score(y_true, y_pred)), "ci95": ci(r2s)},
            "ci":   {"point": float(concordance_index(y_true, y_pred)), "ci95": ci(cis)}}


def score_subset(y_true, y_pred, label, n_boot=1000):
    """Compute metrics + bootstrap CIs for a subset; print and return."""
    if len(y_true) < 5:
        print(f"  {label} (n={len(y_true)}): too small to score"); return None
    m = bootstrap_metrics(y_true, y_pred, n_boot=n_boot)
    print(f"  {label} (n={len(y_true)}):")
    for k in ("ci", "r", "rmse", "r2"):
        b = m[k]; lo, hi = b["ci95"]
        print(f"    {k:<5s} = {b['point']:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    return m


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-samples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2102)  # BAPULM seed
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("BAPULM-STYLE REPLICATION  (ESM-2 650M mean + MolFormer + BAPULM head)")
    print("=" * 80)

    # ── Data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = \
        load_jglaser(args.train_samples)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()
    val_targets_std = (val_targets - t_mean) / t_std

    # ── Embeddings: ESM-2 650M mean-pooled + MolFormer ──
    train_useqs = list(set(zip(train_pn, train_seqs)))
    train_useqs_only = [s for _, s in train_useqs]
    train_unames_only = [n for n, _ in train_useqs]
    train_smi_pairs = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_pairs]
    train_usmiles = [s for _, s in train_smi_pairs]

    print("\n--- Loading ESM-2 650M mean-pooled embeddings ---")
    tr_prot = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                       "jglaser100k", "650M")
    va_prot = extract_esm2_embeddings(val_seqs, val_codes, "pdbbind_val", "650M")
    te_prot = extract_esm2_embeddings(test_seqs, test_codes, "pdbbind_test", "650M")

    print("\n--- Loading MolFormer embeddings ---")
    tr_mol = compute_molformer(train_usmiles, train_ulnames, "jglaser100k")
    va_mol = compute_molformer(val_smiles, val_codes, "pdbbind_val")
    te_mol = compute_molformer(test_smiles, test_codes, "pdbbind_test")

    # ── Random 90/10 split of jglaser-100k (BAPULM uses random_split) ──
    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx, va_idx = perm[:int(0.9 * n)], perm[int(0.9 * n):]

    tr_ds = PairDS([train_pn[i] for i in tr_idx],
                   [train_ln[i] for i in tr_idx],
                   train_targets_std[tr_idx], tr_prot, tr_mol)
    va_ds = PairDS([train_pn[i] for i in va_idx],
                   [train_ln[i] for i in va_idx],
                   train_targets_std[va_idx], tr_prot, tr_mol)
    pdb_te_ds = PairDS(test_codes, test_codes,
                       (test_targets - t_mean) / t_std, te_prot, te_mol)

    train_loader = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(va_ds, args.batch_size, num_workers=2)
    test_loader = DataLoader(pdb_te_ds, args.batch_size, num_workers=2)

    # ── Model ──
    prot_dim = ESM_CONFIGS["650M"]["dim"]   # 1280 (BAPULM uses 1024 from ProtT5)
    mol_dim = 768
    model = BAPULMHead(prot_dim=prot_dim, mol_dim=mol_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {n_params:,}")

    # ── Train (BAPULM recipe) ──
    ckpt = os.path.join(CKPT_DIR, "bapulm_replication.pt")
    print(f"Checkpoint: {ckpt}")

    if os.path.exists(ckpt):
        print(f"  Loaded checkpoint — skipping training.")
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to(DEVICE).eval()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=5)
        criterion = nn.MSELoss()
        best_loss, best_state, wait = float("inf"), None, 0
        import time
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            tl, tn = 0.0, 0
            for p_, m_, y in train_loader:
                p_, m_, y = p_.to(DEVICE), m_.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                pred = model(p_, m_)
                loss = criterion(pred, y)
                loss.backward(); optimizer.step()
                tl += loss.item() * len(y); tn += len(y)
            train_loss = tl / max(tn, 1)

            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for p_, m_, y in val_loader:
                    p_, m_, y = p_.to(DEVICE), m_.to(DEVICE), y.to(DEVICE)
                    pred = model(p_, m_)
                    vl += criterion(pred, y).item() * len(y); vn += len(y)
            val_loss = vl / vn
            scheduler.step(val_loss)
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss; wait = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    [ep {epoch+1:3d}] train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={cur_lr:.1e} best={best_loss:.4f} wait={wait}/{args.patience} "
                  f"({time.time()-t0:.1f}s)" + (" *" if improved else ""), flush=True)
            if wait >= args.patience:
                break
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)
        print(f"  Saved checkpoint {ckpt}")
        model = model.to(DEVICE).eval()

    # ── Predict on PDBbind core / Test2016 ──
    print("\n--- Predicting on Test2016 (290) ---")
    preds = []
    with torch.no_grad():
        for p_, m_, _ in test_loader:
            preds.append(model(p_.to(DEVICE), m_.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean

    print("\nFull Test2016 (n=290):")
    full = score_subset(test_targets, y_pred, "all 290")

    # Leakage-filtered subsets
    leak_path = os.path.join(RESULTS_DIR, "leakage_analysis.json")
    if os.path.exists(leak_path):
        leak = json.load(open(leak_path))
        leaked_pair = set(leak["leaked_pair_codes"])
        leaked_seq  = set(leak["leaked_seq_codes"])
        mask_clean_pair = np.array([c not in leaked_pair for c in test_codes])
        mask_clean_seq  = np.array([c not in leaked_seq  for c in test_codes])
        print("\nLeakage-filtered subsets:")
        clean_pair = score_subset(test_targets[mask_clean_pair], y_pred[mask_clean_pair],
                                   f"CLEAN no-pair-leak (n={mask_clean_pair.sum()})")
        clean_seq  = score_subset(test_targets[mask_clean_seq], y_pred[mask_clean_seq],
                                   f"CLEAN no-seq-leak  (n={mask_clean_seq.sum()})")
    else:
        clean_pair = clean_seq = None

    # ── Evaluate on CSAR-HiQ_36 (BAPULM's only leakage-clean test set) ──
    csar_path = os.path.join(DATA_EXTRA_DIR, "CSAR-HiQ_36.csv")
    csar_results = None
    if os.path.exists(csar_path):
        print("\n--- Predicting on CSAR-HiQ_36 (BAPULM's clean test set) ---")
        import pandas as pd
        cdf = pd.read_csv(csar_path)
        c_codes = [str(p) for p in cdf["pdbid"].tolist()]
        c_seqs  = cdf["seq"].astype(str).tolist()
        c_smis  = cdf["smiles_can"].astype(str).tolist()
        c_aff   = cdf["neg_log10_affinity_M"].to_numpy(dtype=np.float32)

        # ESM-2 650M mean-pool + MolFormer for the 36 CSAR complexes
        c_prot = extract_esm2_embeddings(c_seqs, c_codes, "csar_hiq_36", "650M")
        c_mol  = compute_molformer(c_smis, c_codes, "csar_hiq_36")

        c_ds = PairDS(c_codes, c_codes,
                      (c_aff - t_mean) / t_std, c_prot, c_mol)
        c_loader = DataLoader(c_ds, args.batch_size)
        preds = []
        with torch.no_grad():
            for p_, m_, _ in c_loader:
                preds.append(model(p_.to(DEVICE), m_.to(DEVICE)).cpu())
        c_pred = torch.cat(preds).numpy() * t_std + t_mean
        csar_results = score_subset(c_aff, c_pred, "CSAR-HiQ_36")

        # Save CSAR predictions too
        with open(os.path.join(RESULTS_DIR, "bapulm_replication_csar_predictions.json"), "w") as f:
            json.dump([{"code": c, "target": float(t), "pred": float(p)}
                       for c, t, p in zip(c_codes, c_aff, c_pred)], f, indent=2)
    else:
        print(f"\nNote: {csar_path} not found, skipping CSAR-HiQ_36 evaluation.")

    # ── Save ──
    results = {
        "config": {
            "ligand": "MolFormer",
            "protein_encoder": "ESM-2 650M (mean-pool last layer)",
            "head": "BAPULM (Linear+Linear+BN+Dropout+4Linear)",
            "prot_dim": int(prot_dim),
            "mol_dim": int(mol_dim),
            "n_params": int(n_params),
            "train_samples": int(args.train_samples),
            "epochs_max": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "checkpoint": ckpt,
        },
        "test2016_full":   full,
        "test2016_clean_pair": clean_pair,
        "test2016_clean_seq":  clean_seq,
        "csar_hiq_36":     csar_results,
    }
    out = os.path.join(RESULTS_DIR, "bapulm_replication_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out}")

    pred_out = os.path.join(RESULTS_DIR, "bapulm_replication_predictions.json")
    with open(pred_out, "w") as f:
        json.dump([{"code": c, "target": float(t), "pred": float(yp)}
                   for c, t, yp in zip(test_codes, test_targets, y_pred)],
                  f, indent=2)
    print(f"Saved {pred_out}")

    # ── Scatter ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_targets, y_pred, alpha=0.55, s=25, color="#42A5F5",
               edgecolor="white", linewidth=0.5)
    lo = min(test_targets.min(), y_pred.min()) - 0.3
    hi = max(test_targets.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="red", alpha=0.6, label="y = x")
    ax.set_xlabel("True −log10(Kd/Ki)")
    ax.set_ylabel("Predicted −log10(Kd/Ki)")
    title_r = full["r"]["point"] if full else 0.0
    title_ci = full["ci"]["point"] if full else 0.0
    title_rmse = full["rmse"]["point"] if full else 0.0
    ax.set_title(f"BAPULM-style head — ESM-2 650M + MolFormer\n"
                 f"CI={title_ci:.3f}  r={title_r:.3f}  RMSE={title_rmse:.3f}  (Test2016 n=290)",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "bapulm_replication_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
