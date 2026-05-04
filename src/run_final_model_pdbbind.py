"""
Final model trained on PDBbind refined-minus-core (the standard PDBbind train
set), as opposed to jglaser-100k. Eliminates the jglaser→PDBbind distribution
shift and the string-format leakage concern, since train and test are now
parsed from the same PDB+SDF source via the same BioPython+RDKit pipeline.

Train: PDBbind refined-minus-core (3,767 complexes, random 90/10 split for
       early stopping).
Test:  PDBbind core / CASF-2016 (290) + CSAR-HiQ_36 (36).

Outputs:
  final_pdbbind_results_{LIGAND}_{SETTING}_{ATTN}_{SIZE}.json
  final_pdbbind_predictions_{LIGAND}_{SETTING}_{ATTN}_{SIZE}.json
  final_pdbbind_scatter_{LIGAND}_{SETTING}_{ATTN}_{SIZE}.png
  Checkpoint at /ssd_scratch/akr/checkpoints/final_pdbbind_{tag}.pt
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RESULTS_DIR, FIGURES_DIR, DATA_EXTRA_DIR,
    DEVICE, ESM_CONFIGS, CKPT_DIR,
    load_pdbbind_val_test, extract_esm2_embeddings,
    compute_morgan, compute_chemberta, compute_molformer,
    LayerResidueDS, ResidueDS, LayerDS,
    layer_residue_collate_fn, residue_collate_fn,
    AttnMLP, LayerWMLP, LayerWAttnMLP, ATTN_METHODS,
    train_loop, fwd_residue, fwd_layer, fwd_layer_residue,
    evaluate, concordance_index,
)
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


LIGAND_ENCODERS = {
    "Morgan":    lambda smi, nm, tag: compute_morgan(smi, nm),
    "ChemBERTa": lambda smi, nm, tag: compute_chemberta(smi, nm, tag),
    "MolFormer": lambda smi, nm, tag: compute_molformer(smi, nm, tag),
}


def build_model(setting, attn_name, num_layers, embed_dim, ligand_dim):
    if setting == "B":
        return LayerWMLP(num_layers, embed_dim, ligand_dim)
    acls = ATTN_METHODS[attn_name]
    if setting == "A":
        return AttnMLP(acls, embed_dim, ligand_dim)
    if setting == "C":
        return LayerWAttnMLP(acls, num_layers, embed_dim, ligand_dim)
    raise ValueError(f"Unknown setting {setting}")


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
    def ci(arr): return (float(np.percentile(arr, 2.5)),
                          float(np.percentile(arr, 97.5)))
    return {"r": {"point": float(stats.pearsonr(y_true, y_pred)[0]), "ci95": ci(rs)},
            "rmse": {"point": float(np.sqrt(mean_squared_error(y_true, y_pred))), "ci95": ci(rmses)},
            "r2": {"point": float(r2_score(y_true, y_pred)), "ci95": ci(r2s)},
            "ci": {"point": float(concordance_index(y_true, y_pred)), "ci95": ci(cis)}}


def predict(model, loader, setting):
    preds = []
    with torch.no_grad():
        for batch in loader:
            if setting == "A":
                r, l, m, _ = batch
                p = model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE))
            elif setting == "B":
                ly, lg, _ = batch
                p = model(ly.to(DEVICE), lg.to(DEVICE))
            else:
                ly, rs, lg, mk, _ = batch
                p = model(ly.to(DEVICE), rs.to(DEVICE),
                          lg.to(DEVICE), mk.to(DEVICE))
            preds.append(p.cpu())
    return torch.cat(preds).numpy()


def make_loaders(codes_tr, codes_va, codes_te, t_mean, t_std,
                  prot_layer, prot_res, lig_dict, num_layers,
                  setting, bs_train, bs_pred, targets_tr, targets_va, targets_te):
    if setting == "A":
        tr = ResidueDS(codes_tr, codes_tr, (targets_tr - t_mean)/t_std, prot_res, lig_dict)
        va = ResidueDS(codes_va, codes_va, (targets_va - t_mean)/t_std, prot_res, lig_dict)
        te = ResidueDS(codes_te, codes_te, (targets_te - t_mean)/t_std, prot_res, lig_dict)
        return (DataLoader(tr, bs_train, shuffle=True, collate_fn=residue_collate_fn),
                DataLoader(va, bs_train, collate_fn=residue_collate_fn),
                DataLoader(te, bs_pred, collate_fn=residue_collate_fn),
                fwd_residue)
    if setting == "B":
        tr = LayerDS(codes_tr, codes_tr, (targets_tr - t_mean)/t_std,
                     prot_layer, lig_dict, num_layers)
        va = LayerDS(codes_va, codes_va, (targets_va - t_mean)/t_std,
                     prot_layer, lig_dict, num_layers)
        te = LayerDS(codes_te, codes_te, (targets_te - t_mean)/t_std,
                     prot_layer, lig_dict, num_layers)
        return (DataLoader(tr, 256, shuffle=True),
                DataLoader(va, 256),
                DataLoader(te, 256),
                fwd_layer)
    # C
    tr = LayerResidueDS(codes_tr, codes_tr, (targets_tr - t_mean)/t_std,
                        prot_layer, prot_res, lig_dict, num_layers)
    va = LayerResidueDS(codes_va, codes_va, (targets_va - t_mean)/t_std,
                        prot_layer, prot_res, lig_dict, num_layers)
    te = LayerResidueDS(codes_te, codes_te, (targets_te - t_mean)/t_std,
                        prot_layer, prot_res, lig_dict, num_layers)
    return (DataLoader(tr, bs_train, shuffle=True, collate_fn=layer_residue_collate_fn),
            DataLoader(va, bs_train, collate_fn=layer_residue_collate_fn),
            DataLoader(te, bs_pred, collate_fn=layer_residue_collate_fn),
            fwd_layer_residue)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ligand", default="Morgan", choices=list(LIGAND_ENCODERS))
    p.add_argument("--setting", default="C", choices=["A", "B", "C"])
    p.add_argument("--attn", default="protq+cross", choices=list(ATTN_METHODS))
    p.add_argument("--esm-size", default="650M", choices=list(ESM_CONFIGS))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bs-train", type=int, default=32)
    p.add_argument("--bs-pred",  type=int, default=8)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Random hold-out fraction of refined-minus-core "
                        "for early stopping. Default 0.1 → 90/10 split.")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tag = f"{args.ligand}_S{args.setting}_{args.attn if args.setting != 'B' else 'na'}_{args.esm_size}"
    print("=" * 80)
    print(f"FINAL MODEL on PDBbind refined-minus-core")
    print(f"  ligand={args.ligand}  setting={args.setting}  "
          f"attn={args.attn if args.setting != 'B' else 'n/a'}  "
          f"esm={args.esm_size}")
    print("=" * 80)

    # ── Data: train = refined-minus-core (split 90/10), test = core ──
    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()

    # Random 90/10 split of refined-minus-core for train/val
    n_pool = len(val_codes)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_pool)
    n_val = int(round(n_pool * args.val_frac))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]
    tr_codes = [val_codes[i] for i in tr_idx]
    tr_seqs  = [val_seqs[i]  for i in tr_idx]
    tr_smis  = [val_smiles[i] for i in tr_idx]
    tr_aff   = val_targets[tr_idx]
    va_codes = [val_codes[i] for i in val_idx]
    va_seqs  = [val_seqs[i]  for i in val_idx]
    va_smis  = [val_smiles[i] for i in val_idx]
    va_aff   = val_targets[val_idx]
    print(f"\nTrain: {len(tr_codes)}   Val (early stopping): {len(va_codes)}   "
          f"Test (CASF-2016): {len(test_codes)}")

    t_mean, t_std = float(tr_aff.mean()), float(tr_aff.std())
    print(f"  Train target stats: mean={t_mean:.4f}  std={t_std:.4f}")

    # ── Embeddings ──
    cfg = ESM_CONFIGS[args.esm_size]
    embed_dim = cfg["dim"]
    num_layers = cfg["layers"] + 1

    # All refined-minus-core proteins are already cached as `pdbbind_val` shards
    # (residue) and tensor (alllayer). Reuse them for both train and val splits.
    print(f"\n--- Loading ESM-2 {args.esm_size} embeddings ---")
    needs_residue = args.setting in ("A", "C")
    needs_layer   = args.setting in ("B", "C")

    if needs_residue:
        prot_res_pool = extract_esm2_embeddings(val_seqs, val_codes,
                                                 "pdbbind_val", args.esm_size,
                                                 per_residue=True)
        prot_res_test = extract_esm2_embeddings(test_seqs, test_codes,
                                                 "pdbbind_test", args.esm_size,
                                                 per_residue=True)
    else:
        prot_res_pool = prot_res_test = None
    if needs_layer:
        prot_layer_pool = extract_esm2_embeddings(val_seqs, val_codes,
                                                   "pdbbind_val", args.esm_size,
                                                   all_layers=True)
        prot_layer_test = extract_esm2_embeddings(test_seqs, test_codes,
                                                   "pdbbind_test", args.esm_size,
                                                   all_layers=True)
    else:
        prot_layer_pool = prot_layer_test = None

    # ── Ligand encoder (built per-set so cache keys are right) ──
    fn = LIGAND_ENCODERS[args.ligand]
    print(f"\n--- Building {args.ligand} ligand features ---")
    # All refined-minus-core ligands keyed by pdb code already exist under
    # cache_name "pdbbind_val"; reuse for train+val.
    lig_pool = fn(val_smiles, val_codes, "pdbbind_val")
    lig_test = fn(test_smiles, test_codes, "pdbbind_test")
    ligand_dim = int(len(next(iter(lig_pool.values()))))
    print(f"  ligand_dim = {ligand_dim}")

    # ── Model ──
    model = build_model(args.setting, args.attn, num_layers, embed_dim, ligand_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params (trainable, ESM-2 frozen): {n_params:,}")

    ckpt = os.path.join(CKPT_DIR, f"final_pdbbind_{tag}.pt")
    print(f"Checkpoint: {ckpt}")

    # ── Loaders ──
    train_loader, val_loader, test_loader, fwd_fn = make_loaders(
        tr_codes, va_codes, test_codes, t_mean, t_std,
        prot_layer_pool, prot_res_pool, lig_pool, num_layers,
        args.setting, args.bs_train, args.bs_pred,
        tr_aff, va_aff, test_targets,
    )
    # For test loader we need different prot_layer/res dicts (pdbbind_test),
    # so build it explicitly:
    if args.setting == "A":
        from utils import ResidueDS
        te_ds = ResidueDS(test_codes, test_codes,
                           (test_targets - t_mean)/t_std, prot_res_test, lig_test)
        test_loader = DataLoader(te_ds, args.bs_pred, collate_fn=residue_collate_fn)
    elif args.setting == "B":
        from utils import LayerDS
        te_ds = LayerDS(test_codes, test_codes,
                         (test_targets - t_mean)/t_std,
                         prot_layer_test, lig_test, num_layers)
        test_loader = DataLoader(te_ds, 256)
    else:
        from utils import LayerResidueDS
        te_ds = LayerResidueDS(test_codes, test_codes,
                                (test_targets - t_mean)/t_std,
                                prot_layer_test, prot_res_test, lig_test,
                                num_layers)
        test_loader = DataLoader(te_ds, args.bs_pred,
                                 collate_fn=layer_residue_collate_fn)

    # ── Train ──
    model = train_loop(model, train_loader, val_loader, fwd_fn,
                        epochs=args.epochs, patience=args.patience, ckpt_path=ckpt)

    # ── Predict on Test2016 ──
    print("\n--- Predicting on Test2016 (CASF-2016, 290) ---")
    y_pred = predict(model, test_loader, args.setting) * t_std + t_mean
    test_metrics = bootstrap_metrics(test_targets, y_pred, n_boot=args.n_bootstrap, seed=args.seed)
    print(f"  Test2016 (n={len(test_targets)}):")
    for k in ("ci", "r", "rmse", "r2"):
        b = test_metrics[k]; lo, hi = b["ci95"]
        print(f"    {k:<5s} = {b['point']:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

    # ── Predict on CSAR-HiQ_36 ──
    csar_path = os.path.join(DATA_EXTRA_DIR, "CSAR-HiQ_36.csv")
    csar_metrics = None
    if os.path.exists(csar_path):
        print("\n--- Predicting on CSAR-HiQ_36 ---")
        cdf = pd.read_csv(csar_path)
        c_codes = [str(x) for x in cdf["pdbid"].tolist()]
        c_seqs  = cdf["seq"].astype(str).tolist()
        c_smis  = cdf["smiles_can"].astype(str).tolist()
        c_aff   = cdf["neg_log10_affinity_M"].to_numpy(dtype=np.float32)

        if needs_residue:
            c_res = extract_esm2_embeddings(c_seqs, c_codes, "csar_hiq_36",
                                             args.esm_size, per_residue=True)
        if needs_layer:
            c_lay = extract_esm2_embeddings(c_seqs, c_codes, "csar_hiq_36",
                                             args.esm_size, all_layers=True)
        c_lig = fn(c_smis, c_codes, "csar_hiq_36")

        if args.setting == "A":
            ds = ResidueDS(c_codes, c_codes,
                           (c_aff - t_mean)/t_std, c_res, c_lig)
            loader = DataLoader(ds, args.bs_pred, collate_fn=residue_collate_fn)
        elif args.setting == "B":
            ds = LayerDS(c_codes, c_codes, (c_aff - t_mean)/t_std,
                         c_lay, c_lig, num_layers)
            loader = DataLoader(ds, 36)
        else:
            ds = LayerResidueDS(c_codes, c_codes,
                                 (c_aff - t_mean)/t_std,
                                 c_lay, c_res, c_lig, num_layers)
            loader = DataLoader(ds, args.bs_pred,
                                 collate_fn=layer_residue_collate_fn)
        c_pred = predict(model, loader, args.setting) * t_std + t_mean
        csar_metrics = bootstrap_metrics(c_aff, c_pred, n_boot=args.n_bootstrap, seed=args.seed)
        print(f"  CSAR-HiQ_36 (n={len(c_aff)}):")
        for k in ("ci", "r", "rmse", "r2"):
            b = csar_metrics[k]; lo, hi = b["ci95"]
            print(f"    {k:<5s} = {b['point']:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    else:
        print(f"\nCSAR CSV not found at {csar_path}, skipping CSAR eval.")

    # ── Save ──
    out = {
        "config": {
            "training_source": "PDBbind refined-minus-core",
            "n_train": len(tr_codes), "n_val": len(va_codes),
            "n_test_pdbbind_core": len(test_codes),
            "ligand": args.ligand, "setting": args.setting,
            "attn": args.attn if args.setting != "B" else None,
            "esm_size": args.esm_size,
            "embed_dim": embed_dim, "ligand_dim": ligand_dim,
            "num_layers": num_layers, "n_params": int(n_params),
            "epochs_max": args.epochs, "patience": args.patience,
            "seed": args.seed, "checkpoint": ckpt,
            "t_mean": t_mean, "t_std": t_std,
        },
        "test2016": test_metrics,
        "csar_hiq_36": csar_metrics,
    }
    out_path = fos.path.join(RESULTS_DIR, "final_pdbbind_results_{tag}.json")
    with open(out_path, "w") as f: json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    pred_out = fos.path.join(RESULTS_DIR, "final_pdbbind_predictions_{tag}.json")
    with open(pred_out, "w") as f:
        json.dump([{"code": c, "target": float(t), "pred": float(p)}
                   for c, t, p in zip(test_codes, test_targets, y_pred)],
                  f, indent=2)
    print(f"Saved {pred_out}")

    # ── Scatter ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_targets, y_pred, alpha=0.55, s=25, color="#26A69A",
               edgecolor="white", linewidth=0.5)
    lo = min(test_targets.min(), y_pred.min()) - 0.3
    hi = max(test_targets.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="red", alpha=0.6, label="y = x")
    ax.set_xlabel("True −log10(Kd/Ki)")
    ax.set_ylabel("Predicted −log10(Kd/Ki)")
    point_r = test_metrics["r"]["point"]
    point_ci = test_metrics["ci"]["point"]
    point_rmse = test_metrics["rmse"]["point"]
    ax.set_title(f"Final model — trained on PDBbind refined-minus-core\n"
                 f"{args.ligand} / Setting {args.setting} / {args.esm_size}  ·  "
                 f"CI={point_ci:.3f}  r={point_r:.3f}  RMSE={point_rmse:.3f}",
                 fontweight="bold", fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    fig_path = fos.path.join(FIGURES_DIR, "final_pdbbind_scatter_{tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
