"""
Final model training (PDF Section 5).

Trains a single best-configuration model end-to-end and reports comprehensive
CASF-2016 metrics with bootstrap 95% CIs. Defaults are picked from the
analysis sweep:

  * ESM-2 650M (best peak CI across all three ligand encoders)
  * Setting C: layer-weighted protein + attention pool + ligand (the most
    expressive of the three settings; combines the layer evidence with the
    residue-level signal)
  * ProtQuerySelfCrossAttn (the attention module used in the residue-level
    interpretability runs — already validated to localize cleanly with
    MolFormer)
  * Morgan ligand encoder (highest peak CI in the layer probing sweep:
    0.6986 vs. 0.6895 ChemBERTa vs. 0.5988 MolFormer)

Train: jglaser/binding_affinity (default 100k samples)
Val:   PDBbind v2016 refined minus core (≈3,767 complexes), used for early
       stopping and LR scheduling.
Test:  CASF-2016 / PDBbind v2016 core (290 complexes).

Outputs:
  final_model_results_{LIGAND}_{ARCH}.json     metrics + bootstrap CIs
  final_model_predictions_{LIGAND}_{ARCH}.json per-complex predictions
  final_model_scatter_{LIGAND}_{ARCH}.png      scatter of pred vs. target

Resumes from /ssd_scratch/akr/checkpoints/final_{LIGAND}_{ARCH}.pt if it
exists.
"""

import os
import sys
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RESULTS_DIR, FIGURES_DIR, DATA_EXTRA_DIR,
    DEVICE, ESM_CONFIGS, CKPT_DIR,
    load_jglaser, load_pdbbind_val_test,
    extract_esm2_embeddings, compute_morgan, compute_chemberta, compute_molformer,
    LayerResidueDS, ResidueDS, LayerDS,
    layer_residue_collate_fn, residue_collate_fn,
    AttnMLP, LayerWMLP, LayerWAttnMLP, ATTN_METHODS,
    train_loop, fwd_residue, fwd_layer, fwd_layer_residue,
    evaluate, concordance_index,
)
from torch.utils.data import DataLoader


LIGAND_ENCODERS = OrderedDict([
    ("Morgan",    lambda smi, nm, tag: compute_morgan(smi, nm)),
    ("ChemBERTa", lambda smi, nm, tag: compute_chemberta(smi, nm, tag)),
    ("MolFormer", lambda smi, nm, tag: compute_molformer(smi, nm, tag)),
])


def build_model(arch_setting, attn_name, num_layers, embed_dim, ligand_dim):
    """Build the model for the chosen Setting (A/B/C) and attention method."""
    if arch_setting == "B":
        return LayerWMLP(num_layers, embed_dim, ligand_dim), "layer"
    if attn_name not in ATTN_METHODS:
        raise ValueError(f"Unknown attention method '{attn_name}'. "
                         f"Choices: {list(ATTN_METHODS)}")
    acls = ATTN_METHODS[attn_name]
    if arch_setting == "A":
        return AttnMLP(acls, embed_dim, ligand_dim), "residue"
    if arch_setting == "C":
        return LayerWAttnMLP(acls, num_layers, embed_dim, ligand_dim), "layer_residue"
    raise ValueError(f"Unknown arch setting '{arch_setting}'. Choose A, B, or C.")


def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=0):
    """Bootstrap 95% CI for r, RMSE, R², CI. Returns dict of (point, lo, hi)."""
    from scipy import stats
    from sklearn.metrics import mean_squared_error, r2_score
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = {
        "r":    float(stats.pearsonr(y_true, y_pred)[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "ci":   float(concordance_index(y_true, y_pred)),
    }

    rs, rmses, r2s, cis = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
            continue
        rs.append(stats.pearsonr(yt, yp)[0])
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        r2s.append(r2_score(yt, yp))
        cis.append(concordance_index(yt, yp))

    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "r":    {"point": point["r"],    "ci95": ci(rs)},
        "rmse": {"point": point["rmse"], "ci95": ci(rmses)},
        "r2":   {"point": point["r2"],   "ci95": ci(r2s)},
        "ci":   {"point": point["ci"],   "ci95": ci(cis)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand", default="Morgan",
                        choices=list(LIGAND_ENCODERS),
                        help="Ligand encoder. Default Morgan (best peak CI).")
    parser.add_argument("--setting", default="C", choices=["A", "B", "C"],
                        help="Architecture setting: A=Attn, B=LayerW, "
                             "C=LayerW+Attn. Default C (most expressive).")
    parser.add_argument("--attn", default="protq+cross",
                        choices=list(ATTN_METHODS),
                        help="Attention method (used for settings A and C). "
                             "Default protq+cross.")
    parser.add_argument("--esm-size", default="650M",
                        choices=list(ESM_CONFIGS),
                        help="ESM-2 size. Default 650M (best peak CI).")
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bs-train", type=int, default=32,
                        help="Train batch size (residue paths are O(L^2); "
                             "32 is the cap for a 10 GB GPU at 650M).")
    parser.add_argument("--bs-pred", type=int, default=8,
                        help="Predict batch size for residue paths.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"{args.ligand}_S{args.setting}_{args.attn if args.setting != 'B' else 'na'}_{args.esm_size}"
    print("=" * 80)
    print(f"FINAL MODEL  |  ligand={args.ligand}  setting={args.setting}  "
          f"attn={args.attn if args.setting != 'B' else 'n/a'}  "
          f"esm={args.esm_size}")
    print("=" * 80)

    # ── Data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = \
        load_jglaser(args.train_samples)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()
    val_targets_std = (val_targets - t_mean) / t_std

    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]

    # ── Embeddings ──
    train_useqs = list(set(zip(train_pn, train_seqs)))
    train_useqs_only = [s for _, s in train_useqs]
    train_unames_only = [n for n, _ in train_useqs]

    train_smi_pairs = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_pairs]
    train_usmiles = [s for _, s in train_smi_pairs]

    print(f"\n--- Building {args.ligand} ligand features ---")
    fn = LIGAND_ENCODERS[args.ligand]
    tr_lig = fn(train_usmiles, train_ulnames, "jglaser100k")
    va_lig = fn(val_smiles, val_codes, "pdbbind_val")
    te_lig = fn(test_smiles, test_codes, "pdbbind_test")
    ligand_dim = int(len(next(iter(tr_lig.values()))))
    print(f"  ligand_dim = {ligand_dim}")

    cfg = ESM_CONFIGS[args.esm_size]
    embed_dim = cfg["dim"]
    num_layers = cfg["layers"] + 1

    needs_residue = args.setting in ("A", "C")
    needs_layer = args.setting in ("B", "C")

    if needs_residue:
        tr_res = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                          "jglaser100k", args.esm_size,
                                          per_residue=True)
        va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                          "pdbbind_val", args.esm_size,
                                          per_residue=True)
        te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                          "pdbbind_test", args.esm_size,
                                          per_residue=True)
    if needs_layer:
        tr_layer = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                           "jglaser100k", args.esm_size,
                                           all_layers=True)
        va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                           "pdbbind_val", args.esm_size,
                                           all_layers=True)
        te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                           "pdbbind_test", args.esm_size,
                                           all_layers=True)

    # ── Datasets / loaders ──
    if args.setting == "A":
        tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                          [train_ln[i] for i in tr_idx],
                          train_targets_std[tr_idx], tr_res, tr_lig)
        va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, va_lig)
        te_ds = ResidueDS(test_codes, test_codes,
                          (test_targets - t_mean) / t_std, te_res, te_lig)
        train_loader = DataLoader(tr_ds, args.bs_train, shuffle=True,
                                   collate_fn=residue_collate_fn)
        val_loader = DataLoader(va_ds, args.bs_train,
                                 collate_fn=residue_collate_fn)
        test_loader = DataLoader(te_ds, args.bs_pred,
                                  collate_fn=residue_collate_fn)
        fwd_fn = fwd_residue
    elif args.setting == "B":
        tr_ds = LayerDS([train_pn[i] for i in tr_idx],
                        [train_ln[i] for i in tr_idx],
                        train_targets_std[tr_idx], tr_layer, tr_lig, num_layers)
        va_ds = LayerDS(val_codes, val_codes, val_targets_std,
                        va_layer, va_lig, num_layers)
        te_ds = LayerDS(test_codes, test_codes,
                        (test_targets - t_mean) / t_std,
                        te_layer, te_lig, num_layers)
        train_loader = DataLoader(tr_ds, 512, shuffle=True)
        val_loader = DataLoader(va_ds, 512)
        test_loader = DataLoader(te_ds, 256)
        fwd_fn = fwd_layer
    else:  # C
        tr_ds = LayerResidueDS([train_pn[i] for i in tr_idx],
                                [train_ln[i] for i in tr_idx],
                                train_targets_std[tr_idx],
                                tr_layer, tr_res, tr_lig, num_layers)
        va_ds = LayerResidueDS(val_codes, val_codes, val_targets_std,
                                va_layer, va_res, va_lig, num_layers)
        te_ds = LayerResidueDS(test_codes, test_codes,
                                (test_targets - t_mean) / t_std,
                                te_layer, te_res, te_lig, num_layers)
        train_loader = DataLoader(tr_ds, args.bs_train, shuffle=True,
                                   collate_fn=layer_residue_collate_fn)
        val_loader = DataLoader(va_ds, args.bs_train,
                                 collate_fn=layer_residue_collate_fn)
        test_loader = DataLoader(te_ds, args.bs_pred,
                                  collate_fn=layer_residue_collate_fn)
        fwd_fn = fwd_layer_residue

    # ── Model + training ──
    model, _ = build_model(args.setting, args.attn, num_layers, embed_dim, ligand_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params (trainable head/attn only, ESM-2 frozen): {n_params:,}")

    ckpt = os.path.join(CKPT_DIR, f"final_{tag}.pt")
    print(f"Checkpoint: {ckpt}")

    model = train_loop(
        model, train_loader, val_loader, fwd_fn,
        epochs=args.epochs, patience=args.patience, ckpt_path=ckpt)

    # ── Predict on Test2016 ──
    print("\n--- Predicting on CASF-2016 ---")
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            if args.setting == "A":
                r, l, m, _ = batch
                p = model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE))
            elif args.setting == "B":
                ly, lg, _ = batch
                p = model(ly.to(DEVICE), lg.to(DEVICE))
            else:
                ly, rs, lg, mk, _ = batch
                p = model(ly.to(DEVICE), rs.to(DEVICE),
                          lg.to(DEVICE), mk.to(DEVICE))
            preds.append(p.cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean

    # ── Metrics with bootstrap 95% CI ──
    point_metrics = evaluate(test_targets, y_pred, name=tag)
    print(f"\nTest2016 point metrics:")
    print(f"  CI    = {point_metrics['ci']:.4f}")
    print(f"  r     = {point_metrics['r']:.4f}")
    print(f"  RMSE  = {point_metrics['rmse']:.4f}")
    print(f"  MSE   = {point_metrics['mse']:.4f}")
    print(f"  R²    = {point_metrics['r2']:.4f}")

    print(f"\nBootstrapping {args.n_bootstrap}× for 95% CIs...")
    boot = bootstrap_metrics(test_targets, y_pred,
                             n_boot=args.n_bootstrap, seed=args.seed)
    for k in ("ci", "r", "rmse", "r2"):
        b = boot[k]
        lo, hi = b["ci95"]
        print(f"  {k:<5s} = {b['point']:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

    # ── Save results ──
    results = {
        "config": {
            "ligand":  args.ligand,
            "setting": args.setting,
            "attn":    args.attn if args.setting != "B" else None,
            "esm_size": args.esm_size,
            "embed_dim": embed_dim,
            "ligand_dim": ligand_dim,
            "num_layers": num_layers,
            "train_samples": int(args.train_samples),
            "n_train_used": int(len(tr_idx)),
            "n_val": int(len(val_codes)),
            "n_test": int(len(test_codes)),
            "epochs_max": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "n_params": int(n_params),
            "checkpoint": ckpt,
        },
        "metrics": point_metrics,
        "bootstrap": boot,
    }
    out = f"final_model_results_{tag}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out}")

    pred_out = f"final_model_predictions_{tag}.json"
    with open(pred_out, "w") as f:
        json.dump([{"code": c, "target": float(t), "pred": float(p)}
                   for c, t, p in zip(test_codes, test_targets, y_pred)],
                  f, indent=2)
    print(f"Saved {pred_out}")

    # ── Scatter plot ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_targets, y_pred, alpha=0.55, s=25, color="#42A5F5",
               edgecolor="white", linewidth=0.5)
    lo = min(test_targets.min(), y_pred.min()) - 0.3
    hi = max(test_targets.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="red", alpha=0.6, label="y = x")
    ax.set_xlabel("True −log10(Kd/Ki)")
    ax.set_ylabel("Predicted −log10(Kd/Ki)")
    ax.set_title(f"Final model — {args.ligand} (Setting {args.setting})\n"
                 f"CI={point_metrics['ci']:.3f}  r={point_metrics['r']:.3f}  "
                 f"RMSE={point_metrics['rmse']:.3f}",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    fig_path = f"final_model_scatter_{tag}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
