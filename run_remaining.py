"""
Runs only the remaining ablation experiments (the prior run hung on 650M attn):
  1. MLP | 650M | attn | Morgan
  2. LayerW+MLP | 650M | mean | Morgan

The 650M mean + residue embeddings are already cached; alllayer must be extracted.

Output: remaining_results.json
"""

import os
import sys
import gc
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def train_loop_verbose(model, train_loader, val_loader, forward_fn,
                       lr=1e-3, epochs=200, patience=20, tag=""):
    """Like train_loop but prints per-epoch progress + timing."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = torch.nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tl, tn = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred, target = forward_fn(model, batch)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(target)
            tn += len(target)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred, target = forward_fn(model, batch)
                vl += criterion(pred, target).item() * len(target)
                vn += len(target)
        val_loss = vl / vn
        train_loss = tl / tn
        scheduler.step(val_loss)
        dt = time.time() - t0

        marker = ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(f"  [{tag}] ep {epoch+1:3d} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | {dt:.1f}s{marker}", flush=True)

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    print(f"  Trained {epoch+1} ep, best val={best_loss:.4f}")
    return model.to(DEVICE).eval()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("REMAINING ABLATION EXPERIMENTS (650M attn + 650M LayerW+MLP)")
    print("=" * 80)

    # ── Load data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = load_jglaser(100000)

    # Sanity checks — protein vs ligand names must not be confused
    bad_pn = [x for x in train_pn[:50] if not x.startswith("prot_")]
    bad_ln = [x for x in train_ln[:50] if not x.startswith("lig_")]
    assert not bad_pn, f"train_pn has non-protein names: {bad_pn[:5]}"
    assert not bad_ln, f"train_ln has non-ligand names: {bad_ln[:5]}"
    print(f"  train_pn sample: {train_pn[:3]}")
    print(f"  train_ln sample: {train_ln[:3]}")

    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()

    val_targets_std = (val_targets - t_mean) / t_std
    test_targets_std = (test_targets - t_mean) / t_std

    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]

    # Unique names for morgan FPs
    train_smi_names = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_names]
    train_usmiles = [s for _, s in train_smi_names]

    train_morgan = compute_morgan(train_usmiles, train_ulnames)
    val_morgan = compute_morgan(val_smiles, val_codes)
    test_morgan = compute_morgan(test_smiles, test_codes)

    # For 650M alllayer extraction we need unique seqs
    train_seq_names = list(set(zip(train_pn, train_seqs)))
    train_unames = [n for n, _ in train_seq_names]
    train_useqs = [s for _, s in train_seq_names]

    label = "650M"
    cfg = ESM_CONFIGS[label]
    embed_dim = cfg["dim"]
    num_layers = cfg["layers"] + 1

    results = []
    BS = 128  # smaller batch → less padding waste + lower prefetch memory

    # ═══ LayerW+MLP | 650M | mean | Morgan ═══
    print("\n" + "=" * 80)
    print("LayerW+MLP | 650M | mean | Morgan")
    print("=" * 80)
    print("  (Extracting 650M all-layer embeddings — this takes a while)")

    tr_layer = extract_esm2_embeddings(train_useqs, train_unames,
                                        "jglaser100k", label, all_layers=True)
    va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                        "pdbbind_val", label, all_layers=True)
    te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                        "pdbbind_test", label, all_layers=True)

    # Filter tr_idx to samples whose protein is actually in the cached dict.
    # Cache may have been built with a slightly different name mapping.
    valid_prots = set(tr_layer.keys())
    valid_ligs = set(train_morgan.keys())
    before = len(tr_idx)
    tr_idx = np.array([i for i in tr_idx
                       if train_pn[i] in valid_prots and train_ln[i] in valid_ligs])
    dropped = before - len(tr_idx)
    print(f"  Filtered tr_idx: kept {len(tr_idx)}/{before} "
          f"(dropped {dropped} samples with missing embeddings)")
    if dropped > before * 0.1:
        print(f"  WARNING: dropped >10% of training samples — embedding cache may "
              f"be stale. Consider re-extracting.")

    name2 = f"LayerW+MLP | {label} | mean | Morgan"
    model = LayerWMLP(num_layers, embed_dim, 2048)
    tr_ds = LayerDS([train_pn[i] for i in tr_idx],
                    [train_ln[i] for i in tr_idx],
                    train_targets_std[tr_idx], tr_layer, train_morgan, num_layers)
    va_ds = LayerDS(val_codes, val_codes, val_targets_std,
                    va_layer, val_morgan, num_layers)
    te_ds = LayerDS(test_codes, test_codes, test_targets_std,
                    te_layer, test_morgan, num_layers)

    model = train_loop_verbose(
        model,
        DataLoader(tr_ds, 512, shuffle=True, num_workers=2),
        DataLoader(va_ds, 512, num_workers=2),
        fwd_layer, patience=20, tag="650M-layerW")

    preds = []
    with torch.no_grad():
        for ly, lg, _ in DataLoader(te_ds, 256):
            preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r2 = evaluate(test_targets, y_pred, name2)
    results.append(r2)
    print(f"  CI={r2['ci']:.4f} | r={r2['r']:.4f} | RMSE={r2['rmse']:.4f}")

    lw = torch.softmax(model.lw.data.cpu(), dim=0).numpy()
    top3 = np.argsort(lw)[-3:][::-1]
    print(f"  Top layers: {', '.join(f'L{i}={lw[i]:.3f}' for i in top3)}")

    with open("remaining_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved remaining_results.json")

    # ═══ Summary ═══
    print("\n" + "=" * 80)
    print("REMAINING RESULTS")
    print("=" * 80)
    for r in results:
        print(f"  {r['name']:<38s} | CI={r['ci']:.4f} | r={r['r']:.4f} | "
              f"RMSE={r['rmse']:.4f}")


if __name__ == "__main__":
    main()
