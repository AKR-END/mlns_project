"""
Extra experiments:
  8M   - all 4 × ChemBERTa = 4 experiments (Morgan already done)
  35M  - all 4 × Morgan + all 4 × ChemBERTa = 8 experiments
  650M - mean × Morgan + mean × ChemBERTa = 2 safe experiments
         then ChemBERTa × {max, attn, LayerW} = 3 risky experiments
         (Morgan versions of max/attn/LayerW excluded for 650M)

Output: extra_results.json (saved incrementally + explicit checkpoint
        before the risky 650M runs)
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
            if method == "mean": pooled[name] = emb.mean(dim=0)
            elif method == "max": pooled[name] = emb.max(dim=0)[0]
    else:
        for name, emb in res_src.items():
            if method == "mean": pooled[name] = emb.mean(dim=0)
            elif method == "max": pooled[name] = emb.max(dim=0)[0]
    return pooled


def run_mean(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
             train_pn, train_ln, tr_idx, val_codes, test_codes,
             train_targets_std, val_targets_std, test_targets_std,
             test_targets, t_mean, t_std, train_useqs, train_unames,
             val_seqs, test_seqs, BS=512):
    tr_mean = extract_esm2_embeddings(train_useqs, train_unames,
                                       "jglaser100k", label)
    va_mean = extract_esm2_embeddings(val_seqs, val_codes, "pdbbind_val", label)
    te_mean = extract_esm2_embeddings(test_seqs, test_codes, "pdbbind_test", label)

    name = f"MLP | {label} | mean | {lig_name}"
    print(f"\n  {name}")
    model = MLPProbe(cfg["dim"] + lig_dim)
    tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                     [train_ln[i] for i in tr_idx],
                     train_targets_std[tr_idx], tr_mean, tr_lig)
    va_ds = ConcatDS(val_codes, val_codes, val_targets_std, va_mean, va_lig)
    model = train_loop(model,
        DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
        DataLoader(va_ds, BS, num_workers=4),
        fwd_concat, patience=20)
    te_ds = ConcatDS(test_codes, test_codes, test_targets_std, te_mean, te_lig)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(te_ds, 256):
            preds.append(model(x.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r = evaluate(test_targets, y_pred, name)
    print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
    del model, tr_mean, va_mean, te_mean
    torch.cuda.empty_cache(); gc.collect()
    return r


def run_max(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
            train_pn, train_ln, tr_idx, val_codes, test_codes,
            train_targets_std, val_targets_std, test_targets_std,
            test_targets, t_mean, t_std, train_useqs, train_unames,
            val_seqs, test_seqs, BS=512):
    tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                      "jglaser100k", label, per_residue=True)
    tr_max = pool_residues(tr_res, "max"); del tr_res; gc.collect()
    va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                      "pdbbind_val", label, per_residue=True)
    va_max = pool_residues(va_res, "max"); del va_res; gc.collect()
    te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                      "pdbbind_test", label, per_residue=True)
    te_max = pool_residues(te_res, "max"); del te_res; gc.collect()

    name = f"MLP | {label} | max | {lig_name}"
    print(f"\n  {name}")
    model = MLPProbe(cfg["dim"] + lig_dim)
    tr_ds = ConcatDS([train_pn[i] for i in tr_idx],
                     [train_ln[i] for i in tr_idx],
                     train_targets_std[tr_idx], tr_max, tr_lig)
    va_ds = ConcatDS(val_codes, val_codes, val_targets_std, va_max, va_lig)
    model = train_loop(model,
        DataLoader(tr_ds, BS, shuffle=True, num_workers=4),
        DataLoader(va_ds, BS, num_workers=4),
        fwd_concat, patience=20)
    te_ds = ConcatDS(test_codes, test_codes, test_targets_std, te_max, te_lig)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(te_ds, 256):
            preds.append(model(x.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r = evaluate(test_targets, y_pred, name)
    print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
    del model, tr_max, va_max, te_max
    torch.cuda.empty_cache(); gc.collect()
    return r


def run_attn(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
             train_pn, train_ln, tr_idx, val_codes, test_codes,
             train_targets_std, val_targets_std, test_targets_std,
             test_targets, t_mean, t_std, train_useqs, train_unames,
             val_seqs, test_seqs):
    tr_res = extract_esm2_embeddings(train_useqs, train_unames,
                                      "jglaser100k", label, per_residue=True)
    va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                      "pdbbind_val", label, per_residue=True)
    te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                      "pdbbind_test", label, per_residue=True)

    name = f"MLP | {label} | attn | {lig_name}"
    print(f"\n  {name}")
    model = AttnMLP(ProteinOnlyAttn, cfg["dim"], lig_dim)
    tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                      [train_ln[i] for i in tr_idx],
                      train_targets_std[tr_idx], tr_res, tr_lig)
    va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, va_lig)
    BS = 128
    tr_loader = DataLoader(tr_ds, BS, shuffle=True,
                           collate_fn=residue_collate_fn,
                           num_workers=4, persistent_workers=True,
                           pin_memory=False, prefetch_factor=2)
    va_loader = DataLoader(va_ds, BS, collate_fn=residue_collate_fn,
                           num_workers=2, persistent_workers=True,
                           pin_memory=False, prefetch_factor=2)
    model = train_loop(model, tr_loader, va_loader, fwd_residue, patience=20)
    te_ds = ResidueDS(test_codes, test_codes, test_targets_std, te_res, te_lig)
    preds = []
    with torch.no_grad():
        for r_, l_, m_, _ in DataLoader(te_ds, BS, collate_fn=residue_collate_fn,
                                         num_workers=2):
            preds.append(model(r_.to(DEVICE), l_.to(DEVICE), m_.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r = evaluate(test_targets, y_pred, name)
    print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")
    del model, tr_res, va_res, te_res, tr_ds, va_ds, te_ds, tr_loader, va_loader
    torch.cuda.empty_cache(); gc.collect()
    return r


def run_layerw(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
               train_pn, train_ln, tr_idx, val_codes, test_codes,
               train_targets_std, val_targets_std, test_targets_std,
               test_targets, t_mean, t_std, train_useqs, train_unames,
               val_seqs, test_seqs, BS=512):
    num_layers = cfg["layers"] + 1
    tr_layer = extract_esm2_embeddings(train_useqs, train_unames,
                                        "jglaser100k", label, all_layers=True)
    va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                        "pdbbind_val", label, all_layers=True)
    te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                        "pdbbind_test", label, all_layers=True)

    name = f"LayerW+MLP | {label} | mean | {lig_name}"
    print(f"\n  {name}")
    model = LayerWMLP(num_layers, cfg["dim"], lig_dim)
    tr_ds = LayerDS([train_pn[i] for i in tr_idx],
                    [train_ln[i] for i in tr_idx],
                    train_targets_std[tr_idx], tr_layer, tr_lig, num_layers)
    va_ds = LayerDS(val_codes, val_codes, val_targets_std,
                    va_layer, va_lig, num_layers)
    model = train_loop(model,
        DataLoader(tr_ds, BS, shuffle=True),
        DataLoader(va_ds, BS),
        fwd_layer, patience=20)
    te_ds = LayerDS(test_codes, test_codes, test_targets_std,
                    te_layer, te_lig, num_layers)
    preds = []
    with torch.no_grad():
        for ly, lg, _ in DataLoader(te_ds, 256):
            preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r = evaluate(test_targets, y_pred, name)
    print(f"    CI={r['ci']:.4f} | r={r['r']:.4f} | RMSE={r['rmse']:.4f}")

    lw = torch.softmax(model.lw.data.cpu(), dim=0).numpy()
    top3 = np.argsort(lw)[-3:][::-1]
    print(f"    Top layers: {', '.join(f'L{i}={lw[i]:.3f}' for i in top3)}")
    del model, tr_layer, va_layer, te_layer
    torch.cuda.empty_cache(); gc.collect()
    return r


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("EXTRA EXPERIMENTS: 35M (all 4 × 2 ligands) + 650M (mean × 2 ligands)")
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
    train_unames = [x for x, _ in train_seq_names]
    train_useqs = [s for _, s in train_seq_names]
    train_smi_names = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [x for x, _ in train_smi_names]
    train_usmiles = [s for _, s in train_smi_names]

    train_morgan = compute_morgan(train_usmiles, train_ulnames)
    val_morgan = compute_morgan(val_smiles, val_codes)
    test_morgan = compute_morgan(test_smiles, test_codes)
    train_cberta = compute_chemberta(train_usmiles, train_ulnames, "jglaser100k")
    val_cberta = compute_chemberta(val_smiles, val_codes, "pdbbind_val")
    test_cberta = compute_chemberta(test_smiles, test_codes, "pdbbind_test")

    LIGS = [
        ("Morgan",    train_morgan, val_morgan, test_morgan, 2048),
        ("ChemBERTa", train_cberta, val_cberta, test_cberta, CHEMBERTA_DIM),
    ]

    shared_kwargs = dict(
        train_pn=train_pn, train_ln=train_ln, tr_idx=tr_idx,
        val_codes=val_codes, test_codes=test_codes,
        train_targets_std=train_targets_std,
        val_targets_std=val_targets_std,
        test_targets_std=test_targets_std,
        test_targets=test_targets, t_mean=t_mean, t_std=t_std,
        train_useqs=train_useqs, train_unames=train_unames,
        val_seqs=val_seqs, test_seqs=test_seqs,
    )

    all_results = []

    def save():
        with open("extra_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ── 8M: ChemBERTa only (Morgan already completed in prior runs) ──
    label = "8M"
    cfg = ESM_CONFIGS[label]
    print(f"\n{'=' * 80}\nESM-2 {label} (ChemBERTa only)\n{'=' * 80}")
    lig_name, tr_lig, va_lig, te_lig, lig_dim = LIGS[1]  # ChemBERTa
    for fn in [run_mean, run_max, run_attn, run_layerw]:
        r = fn(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
               **shared_kwargs)
        all_results.append(r); save()

    # ── 35M: all 4 pooling/probe × 2 ligands ──
    label = "35M"
    cfg = ESM_CONFIGS[label]
    print(f"\n{'=' * 80}\nESM-2 {label}\n{'=' * 80}")
    for lig_name, tr_lig, va_lig, te_lig, lig_dim in LIGS:
        print(f"\n--- {lig_name} ---")
        for fn in [run_mean, run_max, run_attn, run_layerw]:
            r = fn(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
                   **shared_kwargs)
            all_results.append(r); save()

    # ── 650M: mean × 2 ligands (safe) ──
    label = "650M"
    cfg = ESM_CONFIGS[label]
    print(f"\n{'=' * 80}\nESM-2 {label} (mean × 2 ligands)\n{'=' * 80}")
    for lig_name, tr_lig, va_lig, te_lig, lig_dim in LIGS:
        print(f"\n--- {lig_name} ---")
        r = run_mean(label, cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
                     **shared_kwargs)
        all_results.append(r); save()

    # ── 650M ChemBERTa: max, attn, LayerW (risky — save checkpoint first) ──
    save()
    print("\n" + "=" * 80)
    print(f"CHECKPOINT: saved {len(all_results)} results before risky 650M runs")
    print("=" * 80)

    lig_name, tr_lig, va_lig, te_lig, lig_dim = LIGS[1]  # ChemBERTa
    print(f"\n{'=' * 80}\nESM-2 650M risky: max/attn/LayerW × ChemBERTa\n{'=' * 80}")
    for fn in [run_max, run_attn, run_layerw]:
        r = fn("650M", cfg, lig_name, tr_lig, va_lig, te_lig, lig_dim,
               **shared_kwargs)
        all_results.append(r); save()

    print("\n" + "=" * 80)
    print("EXTRA RESULTS")
    print("=" * 80)
    print(f"\n{'Model':<44s} | {'CI':>6s} | {'r':>6s} | {'RMSE':>6s}")
    print("-" * 72)
    for r in all_results:
        print(f"  {r['name']:<42s} | {r['ci']:>6.4f} | "
              f"{r['r']:>6.4f} | {r['rmse']:>6.4f}")
    save()
    print("\nSaved extra_results.json")


if __name__ == "__main__":
    main()
