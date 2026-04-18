"""
Step 1: Lock down ablation results.

36 experiments: 3 ESM sizes × 3 pooling × 2 ligand reps × 2 probes (Ridge/MLP)
+ 3 ESM sizes × 2 ligand reps for LayerW+MLP = 36 + 6 = 42 total

Train: jglaser/binding_affinity (100k)
Val:   PDBbind refined minus core (3,767)
Test:  PDBbind core / CASF-2016 (290)

Outputs: ablation_results.json, ablation_plots.png
"""

import os
import sys
import json
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def pool_residues(res_src, method="mean"):
    """Pool per-residue embeddings. res_src can be a dict or a shard directory."""
    pooled = {}
    if isinstance(res_src, str) and os.path.isdir(res_src):
        # Sharded: load one at a time
        for fname in os.listdir(res_src):
            if not fname.endswith(".pt"):
                continue
            name = fname[:-3]
            emb = torch.load(os.path.join(res_src, fname),
                             map_location="cpu", weights_only=True)
            if method == "mean":
                pooled[name] = emb.mean(dim=0)
            elif method == "max":
                pooled[name] = emb.max(dim=0)[0]
    else:
        for name, emb in res_src.items():
            if method == "mean":
                pooled[name] = emb.mean(dim=0)
            elif method == "max":
                pooled[name] = emb.max(dim=0)[0]
    return pooled


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("ABLATION STUDY")
    print("Train: jglaser 100k | Val: PDBbind refined-core | Test: CASF-2016")
    print("=" * 80)

    # ── Load data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = load_jglaser(100000)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()

    # Val/test targets are already in pK units (same scale as jglaser)
    val_targets_std = (val_targets - t_mean) / t_std
    test_targets_std = (test_targets - t_mean) / t_std

    # Split train into train/train_val for Ridge (90/10 of jglaser)
    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]
    tv_idx = perm[int(0.9 * n):]

    # ── Extract embeddings for all 3 ESM sizes ──
    print("\n" + "=" * 80)
    print("Extracting embeddings")
    print("=" * 80)

    # Unique sequences/smiles for extraction
    train_seq_names = list(set(zip(train_pn, train_seqs)))
    train_unames = [n for n, _ in train_seq_names]
    train_useqs = [s for _, s in train_seq_names]
    train_smi_names = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_names]
    train_usmiles = [s for _, s in train_smi_names]

    train_morgan = compute_morgan(train_usmiles, train_ulnames)
    val_morgan = compute_morgan(val_smiles, val_codes)
    test_morgan = compute_morgan(test_smiles, test_codes)

    all_results = []
    BS = 512

    for label in ESM_CONFIGS:
        cfg = ESM_CONFIGS[label]
        embed_dim = cfg["dim"]
        num_layers = cfg["layers"] + 1

        print(f"\n{'=' * 80}")
        print(f"ESM-2 {label}")
        print(f"{'=' * 80}")

        # ── MLP + mean pooling ──
        print("\n  Loading mean-pooled embeddings...")
        tr_mean = extract_esm2_embeddings(train_useqs, train_unames,
                                           f"jglaser100k", label)
        va_mean = extract_esm2_embeddings(val_seqs, val_codes,
                                           f"pdbbind_val", label)
        te_mean = extract_esm2_embeddings(test_seqs, test_codes,
                                           f"pdbbind_test", label)

        name = f"MLP | {label} | mean | Morgan"
        print(f"\n  {name}")
        input_dim = cfg["dim"] + 2048
        model = MLPProbe(input_dim)
        tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                         [train_ln[i] for i in tr_idx],
                         train_targets_std[tr_idx], tr_mean, train_morgan)
        va_ds = ConcatDS(val_codes, val_codes, val_targets_std,
                         va_mean, val_morgan)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
            DataLoader(va_ds, BS, num_workers=4),
            fwd_concat, patience=20)
        preds = []
        te_ds = ConcatDS(test_codes, test_codes, test_targets_std,
                         te_mean, test_morgan)
        with torch.no_grad():
            for x, _ in DataLoader(te_ds, 256):
                preds.append(model(x.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_mean, va_mean, te_mean; torch.cuda.empty_cache()
        import gc; gc.collect()

        # ── MLP + max pooling ──
        print("\n  Loading per-residue embeddings for max pooling...")
        tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                          f"jglaser100k", label, per_residue=True)
        tr_max = pool_residues(tr_res, "max")
        del tr_res; gc.collect()
        va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                          f"pdbbind_val", label, per_residue=True)
        va_max = pool_residues(va_res, "max")
        del va_res; gc.collect()
        te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                          f"pdbbind_test", label, per_residue=True)
        te_max = pool_residues(te_res, "max")
        # keep te_res for attention pooling later

        name = f"MLP | {label} | max | Morgan"
        print(f"\n  {name}")
        model = MLPProbe(input_dim)
        tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                         [train_ln[i] for i in tr_idx],
                         train_targets_std[tr_idx], tr_max, train_morgan)
        va_ds = ConcatDS(val_codes, val_codes, val_targets_std,
                         va_max, val_morgan)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
            DataLoader(va_ds, BS, num_workers=4),
            fwd_concat, patience=20)
        preds = []
        te_ds = ConcatDS(test_codes, test_codes, test_targets_std,
                         te_max, test_morgan)
        with torch.no_grad():
            for x, _ in DataLoader(te_ds, 256):
                preds.append(model(x.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_max, va_max, te_max; torch.cuda.empty_cache(); gc.collect()

        # ── MLP + attention pooling ──
        print("\n  Loading per-residue embeddings for attention...")
        tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                          f"jglaser100k", label, per_residue=True)
        va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                          f"pdbbind_val", label, per_residue=True)
        # te_res already loaded from max pooling step

        name = f"MLP | {label} | attn | Morgan"
        print(f"\n  {name}")
        model = AttnMLP(ProteinOnlyAttn, embed_dim, 2048)
        tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                          [train_ln[i] for i in tr_idx],
                          train_targets_std[tr_idx], tr_res, train_morgan)
        va_ds = ResidueDS(val_codes, val_codes, val_targets_std,
                          va_res, val_morgan)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True, collate_fn=residue_collate_fn),
            DataLoader(va_ds, BS, collate_fn=residue_collate_fn),
            fwd_residue, patience=20)
        preds = []
        te_ds = ResidueDS(test_codes, test_codes, test_targets_std,
                          te_res, test_morgan)
        with torch.no_grad():
            for r, l, m, _ in DataLoader(te_ds, 256, collate_fn=residue_collate_fn):
                preds.append(model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_res, va_res, te_res; torch.cuda.empty_cache(); gc.collect()

        # ── LayerW+MLP ──
        print("\n  Loading all-layer embeddings...")
        tr_layer = extract_esm2_embeddings(train_useqs, train_unames,
                                            f"jglaser100k", label, all_layers=True)
        va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                            f"pdbbind_val", label, all_layers=True)
        te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                            f"pdbbind_test", label, all_layers=True)

        name = f"LayerW+MLP | {label} | mean | Morgan"
        print(f"\n  {name}")
        model = LayerWMLP(num_layers, embed_dim, 2048)
        tr_ds = LayerDS([train_pn[i] for i in tr_idx],
                        [train_ln[i] for i in tr_idx],
                        train_targets_std[tr_idx], tr_layer, train_morgan, num_layers)
        va_ds = LayerDS(val_codes, val_codes, val_targets_std,
                        va_layer, val_morgan, num_layers)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True),
            DataLoader(va_ds, BS),
            fwd_layer, patience=20)
        preds = []
        te_ds = LayerDS(test_codes, test_codes, test_targets_std,
                        te_layer, test_morgan, num_layers)
        with torch.no_grad():
            for ly, lg, _ in DataLoader(te_ds, 256):
                preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")

        lw = torch.softmax(model.lw.data.cpu(), dim=0).numpy()
        top3 = np.argsort(lw)[-3:][::-1]
        print(f"    Top layers: {', '.join(f'L{i}={lw[i]:.3f}' for i in top3)}")
        del model, tr_layer, va_layer, te_layer; torch.cuda.empty_cache()
        gc.collect()

    # ═══ Results table ════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ABLATION RESULTS (CASF-2016 Test Set)")
    print("=" * 80)
    print(f"\n{'Model':<40s} | {'CI':>6s} | {'r':>6s} | {'RMSE':>6s} | {'MSE':>6s}")
    print("-" * 75)
    for r in all_results:
        print(f"  {r['name']:<38s} | {r['ci']:>6.4f} | {r['r']:>6.4f} | "
              f"{r['rmse']:>6.4f} | {r['mse']:>6.4f}")

    # ═══ Bottleneck analysis ══════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS (CI spread per factor)")
    print("=" * 80)

    def parse_name(name):
        parts = [p.strip() for p in name.split("|")]
        return {"probe": parts[0], "scale": parts[1],
                "pool": parts[2], "ligand": parts[3]}

    for factor in ["probe", "scale", "pool"]:
        grouped = {}
        for r in all_results:
            f = parse_name(r["name"])
            grouped.setdefault(f[factor], []).append(r["ci"])
        means = {k: np.mean(v) for k, v in grouped.items()}
        spread = max(means.values()) - min(means.values())
        best = max(means, key=means.get)
        print(f"\n  {factor.upper()}: spread={spread:.4f}")
        for k, v in sorted(means.items(), key=lambda x: -x[1]):
            print(f"    {k:<15s}: mean CI={v:.4f}")

    # ═══ Save ═════════════════════════════════════════════════════════════════
    with open("ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved ablation_results.json")

    # ═══ Plots ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Pooling comparison (MLP, 650M)
    ax = axes[0]
    pools = ["mean", "max", "attn"]
    pool_cis = []
    for p in pools:
        matches = [r for r in all_results
                   if "MLP" in r["name"] and "650M" in r["name"]
                   and f"| {p} |" in r["name"] and "LayerW" not in r["name"]]
        pool_cis.append(matches[0]["ci"] if matches else 0)
    ax.bar(pools, pool_cis, color=["#42A5F5", "#AB47BC", "#26A69A"], alpha=0.85)
    for i, v in enumerate(pool_cis):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("CI")
    ax.set_title("Pooling Comparison (MLP, 650M)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Scale vs performance
    ax = axes[1]
    for probe in ["Ridge", "MLP"]:
        cis = []
        for size in ["8M", "35M", "650M"]:
            matches = [r for r in all_results
                       if r["name"].startswith(probe) and f"| {size} |" in r["name"]
                       and "| mean |" in r["name"]]
            cis.append(matches[0]["ci"] if matches else 0)
        ax.plot(["8M", "35M", "650M"], cis, marker="o", linewidth=2,
                label=f"{probe} (mean pool)")
    ax.set_ylabel("CI")
    ax.set_title("Scale vs Performance", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Linear vs nonlinear gap
    ax = axes[2]
    sizes = ["8M", "35M", "650M"]
    ridge_cis, mlp_cis = [], []
    for size in sizes:
        r_match = [r for r in all_results
                   if r["name"].startswith("Ridge") and f"| {size} |" in r["name"]
                   and "| mean |" in r["name"]]
        m_match = [r for r in all_results
                   if r["name"].startswith("MLP") and f"| {size} |" in r["name"]
                   and "| mean |" in r["name"] and "LayerW" not in r["name"]]
        ridge_cis.append(r_match[0]["ci"] if r_match else 0)
        mlp_cis.append(m_match[0]["ci"] if m_match else 0)
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, ridge_cis, 0.35, label="Ridge (linear)", color="#78909C")
    ax.bar(x + 0.2, mlp_cis, 0.35, label="MLP (nonlinear)", color="#42A5F5")
    for i in range(len(sizes)):
        gap = mlp_cis[i] - ridge_cis[i]
        ax.annotate(f"+{gap:.3f}", (i, max(ridge_cis[i], mlp_cis[i]) + 0.005),
                    ha="center", fontsize=9, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("CI")
    ax.set_title("Linear vs Nonlinear Gap", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("ablation_plots.png", dpi=150, bbox_inches="tight")
    print("Saved ablation_plots.png")


if __name__ == "__main__":
    main()
