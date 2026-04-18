"""
Runs the 12 ablation experiments using ChemBERTa (384-d) as the ligand rep
instead of Morgan (2048-d). Mirrors run_ablation.py structure.

Output: chemberta_results.json
"""

import os
import sys
import gc
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def pool_residues(res_src, method="mean"):
    pooled = {}
    if isinstance(res_src, str) and os.path.isdir(res_src):
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
    print("ABLATION: ChemBERTa ligand representation")
    print("Train: jglaser 100k | Val: PDBbind refined-core | Test: CASF-2016")
    print("=" * 80)

    train_seqs, train_smiles, train_targets, train_pn, train_ln = load_jglaser(100000)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()
    val_targets_std = (val_targets - t_mean) / t_std
    test_targets_std = (test_targets - t_mean) / t_std

    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]

    train_seq_names = list(set(zip(train_pn, train_seqs)))
    train_unames = [n for n, _ in train_seq_names]
    train_useqs = [s for _, s in train_seq_names]
    train_smi_names = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_names]
    train_usmiles = [s for _, s in train_smi_names]

    # ── ChemBERTa ligand features (shared across all experiments) ──
    train_cberta = compute_chemberta(train_usmiles, train_ulnames, "jglaser100k")
    val_cberta = compute_chemberta(val_smiles, val_codes, "pdbbind_val")
    test_cberta = compute_chemberta(test_smiles, test_codes, "pdbbind_test")

    LIG_DIM = CHEMBERTA_DIM
    all_results = []
    BS = 512

    for label in ESM_CONFIGS:
        cfg = ESM_CONFIGS[label]
        embed_dim = cfg["dim"]
        num_layers = cfg["layers"] + 1

        print(f"\n{'=' * 80}\nESM-2 {label}\n{'=' * 80}")

        # ── MLP + mean ──
        print("\n  Loading mean-pooled embeddings...")
        tr_mean = extract_esm2_embeddings(train_useqs, train_unames,
                                           "jglaser100k", label)
        va_mean = extract_esm2_embeddings(val_seqs, val_codes,
                                           "pdbbind_val", label)
        te_mean = extract_esm2_embeddings(test_seqs, test_codes,
                                           "pdbbind_test", label)

        name = f"MLP | {label} | mean | ChemBERTa"
        print(f"\n  {name}")
        model = MLPProbe(cfg["dim"] + LIG_DIM)
        tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                         [train_ln[i] for i in tr_idx],
                         train_targets_std[tr_idx], tr_mean, train_cberta)
        va_ds = ConcatDS(val_codes, val_codes, val_targets_std, va_mean, val_cberta)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
            DataLoader(va_ds, BS, num_workers=4),
            fwd_concat, patience=20)
        preds = []
        te_ds = ConcatDS(test_codes, test_codes, test_targets_std, te_mean, test_cberta)
        with torch.no_grad():
            for x, _ in DataLoader(te_ds, 256):
                preds.append(model(x.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_mean, va_mean, te_mean
        torch.cuda.empty_cache(); gc.collect()

        # ── MLP + max ──
        print("\n  Loading per-residue embeddings for max pooling...")
        tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                          "jglaser100k", label, per_residue=True)
        tr_max = pool_residues(tr_res, "max"); del tr_res; gc.collect()
        va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                          "pdbbind_val", label, per_residue=True)
        va_max = pool_residues(va_res, "max"); del va_res; gc.collect()
        te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                          "pdbbind_test", label, per_residue=True)
        te_max = pool_residues(te_res, "max")

        name = f"MLP | {label} | max | ChemBERTa"
        print(f"\n  {name}")
        model = MLPProbe(cfg["dim"] + LIG_DIM)
        tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                         [train_ln[i] for i in tr_idx],
                         train_targets_std[tr_idx], tr_max, train_cberta)
        va_ds = ConcatDS(val_codes, val_codes, val_targets_std, va_max, val_cberta)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
            DataLoader(va_ds, BS, num_workers=4),
            fwd_concat, patience=20)
        preds = []
        te_ds = ConcatDS(test_codes, test_codes, test_targets_std, te_max, test_cberta)
        with torch.no_grad():
            for x, _ in DataLoader(te_ds, 256):
                preds.append(model(x.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_max, va_max, te_max
        torch.cuda.empty_cache(); gc.collect()

        # ── MLP + attn ──
        print("\n  Loading per-residue embeddings for attention...")
        tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                          "jglaser100k", label, per_residue=True)
        va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                          "pdbbind_val", label, per_residue=True)
        te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                          "pdbbind_test", label, per_residue=True)

        name = f"MLP | {label} | attn | ChemBERTa"
        print(f"\n  {name}")
        model = AttnMLP(ProteinOnlyAttn, embed_dim, LIG_DIM)
        tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                          [train_ln[i] for i in tr_idx],
                          train_targets_std[tr_idx], tr_res, train_cberta)
        va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, val_cberta)
        # Workers to hide sharded I/O, low prefetch to stay under memory
        tr_loader = DataLoader(tr_ds, 128, shuffle=True,
                               collate_fn=residue_collate_fn,
                               num_workers=4, persistent_workers=True,
                               pin_memory=False, prefetch_factor=2)
        va_loader = DataLoader(va_ds, 128, collate_fn=residue_collate_fn,
                               num_workers=2, persistent_workers=True,
                               pin_memory=False, prefetch_factor=2)
        model = train_loop(model, tr_loader, va_loader, fwd_residue, patience=20)
        preds = []
        te_ds = ResidueDS(test_codes, test_codes, test_targets_std,
                          te_res, test_cberta)
        with torch.no_grad():
            for r_, l_, m_, _ in DataLoader(te_ds, 128, collate_fn=residue_collate_fn,
                                             num_workers=2):
                preds.append(model(r_.to(DEVICE), l_.to(DEVICE), m_.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
        del model, tr_res, va_res, te_res, tr_ds, va_ds, te_ds, tr_loader, va_loader
        torch.cuda.empty_cache(); gc.collect()

        # ── LayerW+MLP ──
        print("\n  Loading all-layer embeddings...")
        tr_layer = extract_esm2_embeddings(train_useqs, train_unames,
                                            "jglaser100k", label, all_layers=True)
        va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                            "pdbbind_val", label, all_layers=True)
        te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                            "pdbbind_test", label, all_layers=True)

        name = f"LayerW+MLP | {label} | mean | ChemBERTa"
        print(f"\n  {name}")
        model = LayerWMLP(num_layers, embed_dim, LIG_DIM)
        tr_ds = LayerDS([train_pn[i] for i in tr_idx],
                        [train_ln[i] for i in tr_idx],
                        train_targets_std[tr_idx], tr_layer, train_cberta, num_layers)
        va_ds = LayerDS(val_codes, val_codes, val_targets_std,
                        va_layer, val_cberta, num_layers)
        model = train_loop(
            model,
            DataLoader(tr_ds, BS, shuffle=True),
            DataLoader(va_ds, BS),
            fwd_layer, patience=20)
        preds = []
        te_ds = LayerDS(test_codes, test_codes, test_targets_std,
                        te_layer, test_cberta, num_layers)
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
        del model, tr_layer, va_layer, te_layer
        torch.cuda.empty_cache(); gc.collect()

        # Save incrementally
        with open("chemberta_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("CHEMBERTA RESULTS")
    print("=" * 80)
    print(f"\n{'Model':<40s} | {'CI':>6s} | {'r':>6s} | {'RMSE':>6s}")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['name']:<38s} | {r['ci']:>6.4f} | {r['r']:>6.4f} | {r['rmse']:>6.4f}")

    with open("chemberta_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved chemberta_results.json")


if __name__ == "__main__":
    main()
