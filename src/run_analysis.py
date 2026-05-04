"""
Steps 2-4: Layer analysis, residue-level interpretability, pooling/length analysis,
and 4×3 attention architecture comparison.

Train: jglaser/binding_affinity (100k)
Val:   PDBbind refined minus core (3,767)
Test:  PDBbind core / CASF-2016 (290)

Outputs:
  - layer_analysis.png + layer_results.json     (Step 2)
  - interpretability.png + interp_results.json  (Step 3)
  - length_analysis.png                         (Step 4)
  - attention_comparison.png + attn_results.json (Attention architectures)
"""

import os
import sys
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LAYER-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_layer_analysis(train_pn, train_ln, train_targets_std, tr_idx,
                       train_useqs_only, train_unames_only,
                       val_seqs, val_codes, val_targets_std, val_lig,
                       test_seqs,
                       test_codes, test_targets, test_lig,
                       t_mean, t_std, train_lig, ligand_name="Morgan"):
    """Per-layer probes for all 3 ESM sizes with a given ligand rep."""

    print("\n" + "=" * 80)
    print(f"STEP 2: LAYER-LEVEL ANALYSIS  |  ligand = {ligand_name}")
    print("=" * 80)

    layer_results = {}
    layer_summary = {}

    for label in ESM_CONFIGS:
        cfg = ESM_CONFIGS[label]
        num_layers = cfg["layers"] + 1

        print(f"\n--- ESM-2 {label} ({num_layers} layers) ---")

        # Load/extract all-layer embeddings for this scale.
        # Passing real sequence/name pairs makes this robust when cache is absent.
        tr_alllayer = extract_esm2_embeddings(
            train_useqs_only, train_unames_only, f"jglaser100k", label, all_layers=True)
        va_alllayer = extract_esm2_embeddings(
            val_seqs, val_codes, f"pdbbind_val", label, all_layers=True)
        te_alllayer = extract_esm2_embeddings(
            test_seqs, test_codes, f"pdbbind_test", label, all_layers=True)

        layer_cis = []
        for l in range(num_layers):
            # Extract single-layer mean-pooled
            tr_l = {n: d[l] for n, d in tr_alllayer.items()}
            va_l = {n: d[l] for n, d in va_alllayer.items()}
            te_l = {n: d[l] for n, d in te_alllayer.items()}

            # Ridge probe per layer
            X_tr = []
            for i in tr_idx:
                p = tr_l[train_pn[i]]
                if isinstance(p, torch.Tensor): p = p.numpy()
                lg = train_lig[train_ln[i]]
                X_tr.append(np.concatenate([p, lg]))
            X_tr = np.array(X_tr)

            X_te = []
            for code in test_codes:
                p = te_l[code]
                if isinstance(p, torch.Tensor): p = p.numpy()
                lg = test_lig[code]
                X_te.append(np.concatenate([p, lg]))
            X_te = np.array(X_te)

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X_tr, train_targets_std[tr_idx])
            y_pred = ridge.predict(X_te) * t_std + t_mean
            ci = concordance_index(test_targets, y_pred)
            layer_cis.append(ci)

        layer_results[label] = layer_cis
        peak = np.argmax(layer_cis)
        layer_summary[label] = {
            "peak_layer": int(peak),
            "peak_ci": float(layer_cis[peak]),
            "spread_var": float(np.var(layer_cis)),
            "spread_std": float(np.std(layer_cis)),
            "num_layers": int(num_layers),
        }
        print(f"  Peak layer: L{peak} (CI={layer_cis[peak]:.4f})")
        print(f"  Spread (var): {np.var(layer_cis):.6f}")
        print(f"  Spread (std): {np.std(layer_cis):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"8M": "#66BB6A", "35M": "#FFA726", "650M": "#EF5350"}
    for label, cis in layer_results.items():
        layers = list(range(len(cis)))
        ax.plot(layers, cis, marker="o", markersize=4, linewidth=2,
                color=colors[label], label=f"ESM-2 {label}")
        peak = np.argmax(cis)
        ax.annotate(f"L{peak}={cis[peak]:.3f}",
                    (peak, cis[peak]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.set_xlabel("ESM-2 Layer")
    ax.set_ylabel("CI (Ridge probe)")
    ax.set_title(f"Per-Layer Binding Affinity Signal ({ligand_name})",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = f"layer_analysis_{ligand_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

    res_path = f"layer_results_{ligand_name}.json"
    with open(res_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in layer_results.items()}, f, indent=2)
    print(f"Saved {res_path}")

    sum_path = f"layer_summary_{ligand_name}.json"
    with open(sum_path, "w") as f:
        json.dump(layer_summary, f, indent=2)
    print(f"Saved {sum_path}")

    return layer_results, layer_summary


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: RESIDUE-LEVEL INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _precision_at_k(labels_bool, scores, k):
    n = len(scores)
    if n == 0:
        return 0.0
    k = int(max(1, min(k, n)))
    top_k = np.argsort(scores)[-k:]
    return float(labels_bool[top_k].sum() / k)


def _plot_sequence_attention_heatmaps(samples, out_path):
    if len(samples) == 0:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.2 * n, 4.0)), squeeze=False)
    axes = axes[:, 0]

    for ax, s in zip(axes, samples):
        attn = np.asarray(s["attn"], dtype=np.float32)
        pocket_mask = np.asarray(s["pocket_mask"], dtype=bool)
        code = s["code"]
        auc = s["attn_auc"]

        ax.imshow(attn[np.newaxis, :], aspect="auto", cmap="viridis")
        pocket_pos = np.where(pocket_mask)[0]
        if len(pocket_pos) > 0:
            ax.scatter(pocket_pos, np.zeros_like(pocket_pos), color="red", s=8,
                       marker="|", label="Pocket")
        ax.set_yticks([])
        ax.set_ylabel(code, rotation=0, labelpad=24, va="center")
        ax.set_title(f"{code}: attention vs sequence (AUC={auc:.3f})", fontsize=9)

    axes[-1].set_xlabel("Residue index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def _te_res_accessor(te_res):
    """Return (has_code, get_embedding) callables that work whether te_res is
    a dict of tensors or a sharded directory path (one .pt per code)."""
    if isinstance(te_res, str) and os.path.isdir(te_res):
        def has(code):
            return os.path.exists(os.path.join(te_res, f"{code}.pt"))
        def get(code):
            return torch.load(os.path.join(te_res, f"{code}.pt"),
                              map_location="cpu", weights_only=True)
        return has, get
    return (lambda c: c in te_res), (lambda c: te_res[c])


def run_interpretability(model, test_codes, test_seqs, te_res, te_lig,
                         precision_ks=(10, 25, 50), num_heatmaps=6,
                         ligand_name="Morgan"):
    """Pocket overlap + gradient attribution on CASF-2016 test set."""

    print("\n" + "=" * 80)
    print(f"STEP 3: RESIDUE-LEVEL INTERPRETABILITY  |  ligand = {ligand_name}")
    print("=" * 80)

    has_code, get_embedding = _te_res_accessor(te_res)

    enrichments, aurocs, grad_aurocs, attn_grad_corrs = [], [], [], []
    all_attn_pocket, all_attn_nonpocket = [], []
    precision_at_k = {int(k): [] for k in precision_ks}
    per_complex = []

    for i, code in enumerate(test_codes):
        _, pocket_mask = get_pocket_mask(code)
        if pocket_mask is None or not has_code(code):
            continue

        residue_embs = get_embedding(code)
        seq_len = residue_embs.shape[0]
        if len(pocket_mask) > seq_len:
            pocket_mask = pocket_mask[:seq_len]
        elif len(pocket_mask) < seq_len:
            continue

        n_pocket = pocket_mask.sum()
        if n_pocket == 0 or n_pocket == len(pocket_mask):
            continue

        # Attention weights
        embs = residue_embs.unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, seq_len, dtype=torch.bool, device=DEVICE)
        lig = torch.FloatTensor(te_lig[code]).unsqueeze(0).to(DEVICE)
        model.eval()
        _, attn_w = model.attn_module(embs, lig, mask)
        attn_w = attn_w[0].detach().cpu().numpy()

        # Gradient attribution
        embs_g = residue_embs.unsqueeze(0).to(DEVICE).requires_grad_(True)
        pred = model(embs_g, lig, mask)
        pred.backward()
        grad_w = embs_g.grad[0].norm(dim=-1).cpu().numpy()
        grad_w = grad_w / (grad_w.sum() + 1e-10)

        if len(attn_w) != len(pocket_mask):
            continue

        # Metrics
        all_attn_pocket.extend(attn_w[pocket_mask].tolist())
        all_attn_nonpocket.extend(attn_w[~pocket_mask].tolist())

        p_at_pocket = _precision_at_k(pocket_mask, attn_w, int(n_pocket))
        k = n_pocket
        top_k = np.argsort(attn_w)[-k:]
        prec = pocket_mask[top_k].sum() / k
        expected = n_pocket / len(pocket_mask)
        enrichments.append(prec / expected if expected > 0 else 0)

        for pk in precision_ks:
            precision_at_k[int(pk)].append(_precision_at_k(pocket_mask, attn_w, int(pk)))

        attn_auc = _safe_auc(pocket_mask.astype(int), attn_w)
        grad_auc = _safe_auc(pocket_mask.astype(int), grad_w)
        aurocs.append(attn_auc)
        grad_aurocs.append(grad_auc)

        corr, _ = stats.spearmanr(attn_w, grad_w)
        attn_grad_corrs.append(corr)

        per_complex.append({
            "code": code,
            "seq_len": int(seq_len),
            "n_pocket": int(n_pocket),
            "attn_auc": float(attn_auc),
            "grad_auc": float(grad_auc),
            "precision_at_pocket_size": float(p_at_pocket),
            "enrichment": float(enrichments[-1]),
            "attn_grad_spearman": float(corr),
            "attn": attn_w.astype(float).tolist(),
            "pocket_mask": pocket_mask.astype(bool).tolist(),
        })

    n = len(enrichments)
    print(f"\nAnalyzed {n} complexes with pocket annotations")
    if n == 0:
        print("  No complexes with pocket annotations matched — skipping "
              "interpretability metrics/plots for this ligand.")
        empty = {"ligand": ligand_name, "n_complexes": 0,
                 "conclusion": "no pocket-annotated complexes matched"}
        with open(f"interp_results_{ligand_name}.json", "w") as f:
            json.dump(empty, f, indent=2)
        return empty
    print(f"  Enrichment: {np.mean(enrichments):.2f}x +/- {np.std(enrichments):.2f}")
    print(f"  Attn AUC-ROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"  Grad AUC-ROC: {np.mean(grad_aurocs):.4f} +/- {np.std(grad_aurocs):.4f}")
    print(f"  Attn-Grad Spearman: {np.mean(attn_grad_corrs):.4f}")
    for pk in sorted(precision_at_k):
        vals = precision_at_k[pk]
        if len(vals) == 0:
            continue
        print(f"  Precision@{pk}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    pocket_ratio = np.mean(all_attn_pocket) / (np.mean(all_attn_nonpocket) + 1e-10)
    _, p_val = stats.mannwhitneyu(all_attn_pocket, all_attn_nonpocket,
                                   alternative="greater")
    print(f"  Pocket/non-pocket ratio: {pocket_ratio:.2f}x (p={p_val:.2e})")

    if np.mean(enrichments) > 1.5 and np.mean(aurocs) > 0.6:
        conclusion = "Attention LOCALIZES to binding pockets"
    elif np.mean(enrichments) > 1.0:
        conclusion = "Weak but significant pocket localization"
    else:
        conclusion = "Attention does NOT localize to pockets (global encoding)"
    print(f"\n  CONCLUSION: {conclusion}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(enrichments, bins=20, color="#26A69A", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="Random (1.0x)")
    ax.axvline(np.mean(enrichments), color="blue", label=f"Mean ({np.mean(enrichments):.2f}x)")
    ax.set_xlabel("Pocket Enrichment"); ax.set_ylabel("Count")
    ax.set_title("Attention → Pocket Enrichment", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(aurocs, bins=20, alpha=0.7, color="#42A5F5", label="Attention")
    ax.hist(grad_aurocs, bins=20, alpha=0.7, color="#FF7043", label="Gradient")
    ax.axvline(0.5, color="red", linestyle="--")
    ax.set_xlabel("AUC-ROC"); ax.set_ylabel("Count")
    ax.set_title("Pocket Prediction AUC", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, max(np.percentile(all_attn_pocket, 99),
                              np.percentile(all_attn_nonpocket, 99)), 50)
    ax.hist(all_attn_nonpocket, bins=bins, alpha=0.7, density=True,
            color="#78909C", label="Non-pocket")
    ax.hist(all_attn_pocket, bins=bins, alpha=0.7, density=True,
            color="#EF5350", label="Pocket")
    ax.set_xlabel("Attention Weight"); ax.set_ylabel("Density")
    ax.set_title("Pocket vs Non-pocket Attention", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(attn_grad_corrs, bins=20, color="#AB47BC", alpha=0.8)
    ax.axvline(np.mean(attn_grad_corrs), color="blue",
               label=f"Mean ({np.mean(attn_grad_corrs):.3f})")
    ax.set_xlabel("Spearman r (Attn vs Grad)"); ax.set_ylabel("Count")
    ax.set_title("Attention Faithfulness", fontweight="bold")
    ax.legend(fontsize=8)

    plt.suptitle(f"Interpretability [{ligand_name}]: {conclusion}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    interp_fig = f"interpretability_{ligand_name}.png"
    plt.savefig(interp_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {interp_fig}")

    # One strong sequence-level figure: top complexes by attention AUC.
    top_samples = sorted(per_complex, key=lambda x: x["attn_auc"], reverse=True)[:num_heatmaps]
    heatmap_path = f"sequence_attention_heatmaps_{ligand_name}.png"
    _plot_sequence_attention_heatmaps(top_samples, heatmap_path)

    results = {"ligand": ligand_name,
               "enrichment": np.mean(enrichments), "auroc": np.mean(aurocs),
               "grad_auroc": np.mean(grad_aurocs),
               "attn_grad_corr": np.mean(attn_grad_corrs),
               "pocket_ratio": pocket_ratio, "p_value": p_val,
               "conclusion": conclusion,
               "precision_at_k": {
                   str(pk): float(np.mean(vals)) if len(vals) else 0.0
                   for pk, vals in precision_at_k.items()
               }}
    res_path = f"interp_results_{ligand_name}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {res_path}")

    pc_path = f"interp_per_complex_{ligand_name}.json"
    with open(pc_path, "w") as f:
        json.dump(per_complex, f, indent=2)
    print(f"Saved {pc_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: POOLING + LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_length_analysis(mean_model, attn_model, test_codes, test_seqs,
                        test_targets, te_mean, te_res, te_lig,
                        t_mean, t_std):
    """Compare mean vs attention pooling performance by protein length."""

    print("\n" + "=" * 80)
    print("STEP 4: POOLING + LENGTH ANALYSIS")
    print("=" * 80)

    # Predict with mean-pooled model
    te_concat = ConcatDS(test_codes, test_codes,
                         (test_targets - t_mean) / t_std, te_mean, te_lig)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(te_concat, 256):
            preds.append(mean_model(x.to(DEVICE)).cpu())
    y_pred_mean = torch.cat(preds).numpy() * t_std + t_mean

    # Predict with attention model. Self-attention at 650M is O(L^2) in memory,
    # so use a small batch size — long PDBbind proteins blow out a 10 GB GPU
    # at batch=256.
    te_res_ds = ResidueDS(test_codes, test_codes,
                          (test_targets - t_mean) / t_std, te_res, te_lig)
    preds = []
    with torch.no_grad():
        for r, l, m, _ in DataLoader(te_res_ds, 8, collate_fn=residue_collate_fn):
            preds.append(attn_model(r.to(DEVICE), l.to(DEVICE),
                                     m.to(DEVICE)).cpu())
    y_pred_attn = torch.cat(preds).numpy() * t_std + t_mean

    seq_lengths = np.array([len(s) for s in test_seqs])
    bins = [0, 200, 400, 600, 800, 1200]
    bin_labels = ["<200", "200-400", "400-600", "600-800", "800+"]
    bucket_idx = np.clip(np.digitize(seq_lengths, bins) - 1, 0, len(bin_labels) - 1)

    print(f"\n{'Bucket':<10s} | {'N':>4s} | {'CI mean':>8s} | {'CI attn':>8s} | {'Delta':>7s}")
    print("-" * 50)
    bucket_stats = []
    for i, label in enumerate(bin_labels):
        mask = bucket_idx == i
        n = mask.sum()
        if n < 5: continue
        ci_m = concordance_index(test_targets[mask], y_pred_mean[mask])
        ci_a = concordance_index(test_targets[mask], y_pred_attn[mask])
        print(f"  {label:<8s} | {n:>4d} | {ci_m:>8.4f} | {ci_a:>8.4f} | {ci_a - ci_m:>+7.4f}")
        bucket_stats.append({"label": label, "n": n, "ci_mean": ci_m,
                              "ci_attn": ci_a, "delta": ci_a - ci_m})

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [b["label"] for b in bucket_stats]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [b["ci_mean"] for b in bucket_stats], 0.35,
           label="Mean pool", color="#42A5F5")
    ax.bar(x + 0.2, [b["ci_attn"] for b in bucket_stats], 0.35,
           label="Attn pool", color="#26A69A")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Protein Length"); ax.set_ylabel("CI")
    ax.set_title("Performance vs Sequence Length", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("length_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved length_analysis.png")
    return bucket_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4×3 ATTENTION ARCHITECTURE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_attention_comparison(train_pn, train_ln, train_targets_std, tr_idx,
                             val_codes, val_targets_std, val_lig,
                             test_codes, test_targets, test_lig,
                             tr_res, va_res, te_res,
                             tr_layer, va_layer, te_layer,
                             train_lig, t_mean, t_std,
                             ligand_dim, ligand_name="Morgan"):
    """4 attention methods × 3 settings under the selected ligand encoder."""

    print("\n" + "=" * 80)
    print(f"4×3 ATTENTION ARCHITECTURE COMPARISON  |  ligand = {ligand_name}")
    print("=" * 80)

    cfg = ESM_CONFIGS["650M"]
    embed_dim, num_layers = cfg["dim"], cfg["layers"] + 1
    # Residue-level batches at 650M are O(L^2) memory in the self-attention
    # path; a 10 GB GPU OOMs above ~32. Layer-only (Setting B) has no quadratic
    # term so it can stay large.
    BS_RES = 32
    BS_PRED_RES = 8
    BS_LAYER = 512
    BS_PRED_LAYER = 256
    all_results = []

    # ── Setting A: MLP + Attention (4 variants) ──
    print("\n--- Setting A: MLP + Attention ---")
    tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                      [train_ln[i] for i in tr_idx],
                      train_targets_std[tr_idx], tr_res, train_lig)
    va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, val_lig)
    te_ds = ResidueDS(test_codes, test_codes,
                      (test_targets - t_mean) / t_std, te_res, test_lig)

    for aname, acls in ATTN_METHODS.items():
        name = f"A: MLP+{aname}"
        print(f"\n  {name}")
        ckpt = os.path.join(CKPT_DIR, f"attn_A_{aname}_{ligand_name}.pt")
        model = train_loop(
            AttnMLP(acls, embed_dim, ligand_dim),
            DataLoader(tr_ds, BS_RES, shuffle=True, collate_fn=residue_collate_fn),
            DataLoader(va_ds, BS_RES, collate_fn=residue_collate_fn),
            fwd_residue, patience=20, ckpt_path=ckpt)
        preds = []
        with torch.no_grad():
            for r, l, m, _ in DataLoader(te_ds, BS_PRED_RES, collate_fn=residue_collate_fn):
                preds.append(model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f}")
        del model; torch.cuda.empty_cache()

    # ── Setting B: MLP + Layer Weighting (1 variant) ──
    print("\n--- Setting B: MLP + Layer Weighting ---")
    tr_lds = LayerDS([train_pn[i] for i in tr_idx],
                     [train_ln[i] for i in tr_idx],
                     train_targets_std[tr_idx], tr_layer, train_lig, num_layers)
    va_lds = LayerDS(val_codes, val_codes, val_targets_std,
                     va_layer, val_lig, num_layers)
    te_lds = LayerDS(test_codes, test_codes,
                     (test_targets - t_mean) / t_std,
                     te_layer, test_lig, num_layers)

    name = "B: MLP+LayerW"
    print(f"\n  {name}")
    ckpt = os.path.join(CKPT_DIR, f"attn_B_{ligand_name}.pt")
    model = train_loop(
        LayerWMLP(num_layers, embed_dim, ligand_dim),
        DataLoader(tr_lds, BS_LAYER, shuffle=True),
        DataLoader(va_lds, BS_LAYER),
        fwd_layer, patience=20, ckpt_path=ckpt)
    preds = []
    with torch.no_grad():
        for ly, lg, _ in DataLoader(te_lds, BS_PRED_LAYER):
            preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * t_std + t_mean
    r = evaluate(test_targets, y_pred, name)
    all_results.append(r)
    print(f"    CI={r['ci']:.4f} | r={r['r']:.4f}")
    del model; torch.cuda.empty_cache()

    # ── Setting C: MLP + LayerW + Attention (4 variants) ──
    print("\n--- Setting C: MLP + LayerW + Attention ---")
    tr_lrs = LayerResidueDS([train_pn[i] for i in tr_idx],
                            [train_ln[i] for i in tr_idx],
                            train_targets_std[tr_idx],
                            tr_layer, tr_res, train_lig, num_layers)
    va_lrs = LayerResidueDS(val_codes, val_codes, val_targets_std,
                            va_layer, va_res, val_lig, num_layers)
    te_lrs = LayerResidueDS(test_codes, test_codes,
                            (test_targets - t_mean) / t_std,
                            te_layer, te_res, test_lig, num_layers)

    for aname, acls in ATTN_METHODS.items():
        name = f"C: MLP+LayerW+{aname}"
        print(f"\n  {name}")
        ckpt = os.path.join(CKPT_DIR, f"attn_C_{aname}_{ligand_name}.pt")
        model = train_loop(
            LayerWAttnMLP(acls, num_layers, embed_dim, ligand_dim),
            DataLoader(tr_lrs, BS_RES, shuffle=True, collate_fn=layer_residue_collate_fn),
            DataLoader(va_lrs, BS_RES, collate_fn=layer_residue_collate_fn),
            fwd_layer_residue, patience=20, ckpt_path=ckpt)
        preds = []
        with torch.no_grad():
            for ly, rs, lg, mk, _ in DataLoader(
                    te_lrs, BS_PRED_RES, collate_fn=layer_residue_collate_fn):
                preds.append(model(ly.to(DEVICE), rs.to(DEVICE),
                                   lg.to(DEVICE), mk.to(DEVICE)).cpu())
        y_pred = torch.cat(preds).numpy() * t_std + t_mean
        r = evaluate(test_targets, y_pred, name)
        all_results.append(r)
        print(f"    CI={r['ci']:.4f} | r={r['r']:.4f}")
        del model; torch.cuda.empty_cache()

    # Results table
    print(f"\n{'Model':<30s} | {'CI':>6s} | {'r':>6s} | {'RMSE':>6s}")
    print("-" * 55)
    for r in all_results:
        print(f"  {r['name']:<28s} | {r['ci']:>6.4f} | {r['r']:>6.4f} | {r['rmse']:>6.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_s = {"A": "#42A5F5", "B": "#66BB6A", "C": "#AB47BC"}
    x_pos, xticks, xlabels = 0, [], []
    for setting in ["A", "B", "C"]:
        matches = [r for r in all_results if r["name"].startswith(setting)]
        for r in matches:
            c = colors_s[setting]
            ax.bar(x_pos, r["ci"], color=c, alpha=0.85, edgecolor="white")
            ax.text(x_pos, r["ci"] + 0.001, f"{r['ci']:.3f}", ha="center",
                    fontsize=7, rotation=90)
            short = r["name"].split(":")[-1].strip()
            xticks.append(x_pos); xlabels.append(short)
            x_pos += 1
        x_pos += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("CI")
    ax.set_title(f"4 Attention Methods × 3 Settings  (ligand={ligand_name})",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"Setting {s}")
                       for s, c in colors_s.items()], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = f"attention_comparison_{ligand_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")

    res_path = f"attn_results_{ligand_name}.json"
    with open(res_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {res_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100000,
                        help="Number of jglaser training samples to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-length-analysis", action="store_true",
                        help="Also run pooling/length analysis (Step 4)")
    parser.add_argument("--run-attention-comparison", action="store_true",
                        help="Also run 4x3 attention architecture comparison")
    parser.add_argument("--precision-ks", type=str, default="10,25,50",
                        help="Comma-separated K values for Precision@K")
    parser.add_argument("--num-heatmaps", type=int, default=6,
                        help="Number of sequence heatmaps to plot")
    parser.add_argument("--ligand", type=str, default=None,
                        choices=["Morgan", "ChemBERTa", "MolFormer"],
                        help="Run only one ligand encoder (default: all 3). "
                             "Useful for parallelizing across jobs.")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the per-ligand layer + interpretability "
                             "sweep (Steps 2 and 3). Use when those outputs "
                             "already exist and you only want to run the "
                             "length-analysis or attention-comparison branches.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    precision_ks = tuple(int(x.strip()) for x in args.precision_ks.split(",") if x.strip())

    print("=" * 80)
    print("ANALYSIS: Layer-level + Residue-level interpretability")
    print(f"Train: jglaser {args.train_samples} | Val: PDBbind refined-core | Test: CASF-2016")
    print("=" * 80)

    # ── Load data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = load_jglaser(args.train_samples)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()
    val_targets_std = (val_targets - t_mean) / t_std

    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]

    # ── Extract embeddings (650M for steps 3-4 and attention comparison) ──
    train_useqs = list(set(zip(train_pn, train_seqs)))
    train_useqs_only = [s for _, s in train_useqs]
    train_unames_only = [n for n, _ in train_useqs]

    train_smi_pairs = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_pairs]
    train_usmiles = [s for _, s in train_smi_pairs]

    # Build ligand encoders. By default sweep all 3; if --ligand is given,
    # build only that one (lets you split into 3 parallel jobs).
    ligand_encoders = OrderedDict([
        ("Morgan",    lambda smi, nm, tag: compute_morgan(smi, nm)),
        ("ChemBERTa", lambda smi, nm, tag: compute_chemberta(smi, nm, tag)),
        ("MolFormer", lambda smi, nm, tag: compute_molformer(smi, nm, tag)),
    ])
    if args.ligand is not None:
        ligand_encoders = OrderedDict([(args.ligand, ligand_encoders[args.ligand])])
        print(f"\n>>> Running single ligand: {args.ligand}")

    ligand_dicts = {}  # {lig_name: (train_dict, val_dict, test_dict, dim)}
    for lig_name, fn in ligand_encoders.items():
        print(f"\n--- Building {lig_name} ligand features ---")
        tr = fn(train_usmiles, train_ulnames, "jglaser100k")
        va = fn(val_smiles, val_codes, "pdbbind_val")
        te = fn(test_smiles, test_codes, "pdbbind_test")
        any_vec = next(iter(tr.values()))
        ligand_dicts[lig_name] = (tr, va, te, int(len(any_vec)))

    # Pick the primary ligand for the optional architecture sweeps
    # (length analysis, 4×3 attention comparison). Defaults to Morgan when
    # all three were built; otherwise falls back to whichever was selected.
    primary_lig = "Morgan" if "Morgan" in ligand_dicts else next(iter(ligand_dicts))
    pri_train_lig, pri_val_lig, pri_test_lig, pri_lig_dim = ligand_dicts[primary_lig]
    print(f"\nPrimary ligand for length / attention-comparison branches: "
          f"{primary_lig} ({pri_lig_dim}-d)")

    # 650M per-residue embeddings (sharded dir paths, not loaded into RAM).
    tr_res = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                      "jglaser100k", "650M", per_residue=True)
    va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                      "pdbbind_val", "650M", per_residue=True)
    te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                      "pdbbind_test", "650M", per_residue=True)

    # The mean- and all-layer 650M dicts are heavy (multi-GB) and only used by
    # the optional length / attention-comparison branches. Load lazily below.
    tr_mean = te_mean = None
    tr_layer = va_layer = te_layer = None

    # ── Sweep: layer + residue-level analyses across all 3 ligand encoders ──
    cfg = ESM_CONFIGS["650M"]
    layer_all = {}        # {lig_name: {esm_size: [per-layer CIs]}}
    layer_summary_all = {}
    interp_all = {}
    trained_attn_models = {}

    if args.skip_base:
        print("\n>>> --skip-base: skipping per-ligand layer + interpretability "
              "sweep (Steps 2 and 3).")
    else:
        for lig_name, (tr_lig, va_lig, te_lig, lig_dim) in ligand_dicts.items():
            print(f"\n{'#' * 80}\n# LIGAND SWEEP: {lig_name} ({lig_dim}-d)\n{'#' * 80}")

            # Step 2: per-layer Ridge probes
            lr, ls = run_layer_analysis(
                train_pn, train_ln, train_targets_std, tr_idx,
                train_useqs_only, train_unames_only,
                val_seqs, val_codes, val_targets_std, va_lig,
                test_seqs,
                test_codes, test_targets, te_lig,
                t_mean, t_std, tr_lig, ligand_name=lig_name)
            layer_all[lig_name] = {k: [float(x) for x in v] for k, v in lr.items()}
            layer_summary_all[lig_name] = ls

            # Step 3: train 650M attention model with this ligand, then interpret
            print(f"\nTraining attention model ({lig_name}) for interpretability...")
            tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                              [train_ln[i] for i in tr_idx],
                              train_targets_std[tr_idx], tr_res, tr_lig)
            va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, va_lig)
            ckpt = os.path.join(CKPT_DIR, f"interp_attn_{lig_name}.pt")
            attn_model = train_loop(
                AttnMLP(ProtQuerySelfCrossAttn, cfg["dim"], lig_dim),
                DataLoader(tr_ds, 32, shuffle=True, collate_fn=residue_collate_fn),
                DataLoader(va_ds, 32, collate_fn=residue_collate_fn),
                fwd_residue, patience=20, ckpt_path=ckpt)

            interp_all[lig_name] = run_interpretability(
                attn_model, test_codes, test_seqs, te_res, te_lig,
                precision_ks=precision_ks,
                num_heatmaps=args.num_heatmaps,
                ligand_name=lig_name)
            trained_attn_models[lig_name] = attn_model

    # Aggregate plots/files only when all 3 ligands ran in this invocation;
    # otherwise leave the per-ligand files alone (they can be combined later).
    if len(layer_all) == 3:
        with open("layer_results_all.json", "w") as f:
            json.dump(layer_all, f, indent=2)
        with open("layer_summary_all.json", "w") as f:
            json.dump(layer_summary_all, f, indent=2)
        with open("interp_results_all.json", "w") as f:
            json.dump({k: {kk: (vv if not isinstance(vv, dict) else vv)
                           for kk, vv in v.items()} for k, v in interp_all.items()},
                      f, indent=2, default=float)
        print("Saved layer_results_all.json, layer_summary_all.json, interp_results_all.json")

        # Overlay plot: CI vs layer, one subplot per ESM size, one line per ligand
        sizes = list(ESM_CONFIGS.keys())
        fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.5),
                                 squeeze=False)
        lig_colors = {"Morgan": "#42A5F5", "ChemBERTa": "#EF5350",
                      "MolFormer": "#66BB6A"}
        for ax, size in zip(axes[0], sizes):
            for lig_name in layer_all:
                cis = layer_all[lig_name][size]
                ax.plot(range(len(cis)), cis, marker="o", markersize=3,
                        linewidth=1.8, label=lig_name,
                        color=lig_colors.get(lig_name))
            ax.set_title(f"ESM-2 {size}", fontweight="bold")
            ax.set_xlabel("Layer"); ax.set_ylabel("CI (Ridge)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("layer_analysis_all_ligands.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved layer_analysis_all_ligands.png")

    # Keep an attention model around for optional length analysis. If the base
    # sweep was skipped (--skip-base), train a single attention model on the
    # primary ligand only so the length-analysis branch has something to score.
    if trained_attn_models:
        attn_model = trained_attn_models.get(primary_lig,
                                             next(iter(trained_attn_models.values())))
    elif args.run_length_analysis:
        print(f"\n>>> --skip-base + --run-length-analysis: training a fresh "
              f"650M attention model on {primary_lig} for length scoring "
              f"(this is the only model needed for length analysis).")
        tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                          [train_ln[i] for i in tr_idx],
                          train_targets_std[tr_idx], tr_res, pri_train_lig)
        va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, pri_val_lig)
        # Reuse the same checkpoint key as the per-ligand sweep so length-
        # analysis runs with --skip-base reuse the model when one exists.
        ckpt = os.path.join(CKPT_DIR, f"interp_attn_{primary_lig}.pt")
        attn_model = train_loop(
            AttnMLP(ProtQuerySelfCrossAttn, cfg["dim"], pri_lig_dim),
            DataLoader(tr_ds, 32, shuffle=True, collate_fn=residue_collate_fn),
            DataLoader(va_ds, 32, collate_fn=residue_collate_fn),
            fwd_residue, patience=20, ckpt_path=ckpt)
    else:
        attn_model = None  # not needed for the attention-comparison branch

    # Optional extras for deeper architecture analysis.
    # Both branches use the primary ligand (selected via --ligand or Morgan).
    if args.run_length_analysis:
        print(f"\n>>> Length analysis using ligand={primary_lig} "
              f"({pri_lig_dim}-d)")
        tr_mean = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                          "jglaser100k", "650M")
        te_mean = extract_esm2_embeddings(test_seqs, test_codes,
                                          "pdbbind_test", "650M")
        va_mean = extract_esm2_embeddings(val_seqs, val_codes, "pdbbind_val", "650M")
        va_concat = ConcatDS(val_codes, val_codes, val_targets_std, va_mean, pri_val_lig)
        tr_concat = ConcatDS([train_pn[i] for i in tr_idx],
                             [train_ln[i] for i in tr_idx],
                             train_targets_std[tr_idx], tr_mean, pri_train_lig)
        ckpt = os.path.join(CKPT_DIR, f"length_meanpool_{primary_lig}.pt")
        mean_model = train_loop(
            MLPProbe(cfg["dim"] + pri_lig_dim),
            DataLoader(tr_concat, 512, shuffle=True, num_workers=4),
            DataLoader(va_concat, 512, num_workers=4),
            fwd_concat, patience=20, ckpt_path=ckpt)

        # Use the attention model trained for the primary ligand if available.
        primary_attn_model = trained_attn_models.get(primary_lig, attn_model)
        run_length_analysis(mean_model, primary_attn_model, test_codes, test_seqs,
                            test_targets, te_mean, te_res, pri_test_lig, t_mean, t_std)
        del mean_model, tr_mean, te_mean, va_mean
        torch.cuda.empty_cache()

    if args.run_attention_comparison:
        print(f"\n>>> Attention comparison using ligand={primary_lig} "
              f"({pri_lig_dim}-d)")
        tr_layer = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                           "jglaser100k", "650M", all_layers=True)
        va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                           "pdbbind_val", "650M", all_layers=True)
        te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                           "pdbbind_test", "650M", all_layers=True)
        run_attention_comparison(
            train_pn, train_ln, train_targets_std, tr_idx,
            val_codes, val_targets_std, pri_val_lig,
            test_codes, test_targets, pri_test_lig,
            tr_res, va_res, te_res, tr_layer, va_layer, te_layer,
            pri_train_lig, t_mean, t_std,
            ligand_dim=pri_lig_dim, ligand_name=primary_lig)

    if attn_model is not None:
        del attn_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Outputs:")
    print("  - layer_analysis.png + layer_results.json + layer_summary.json")
    print("  - interpretability.png + interp_results.json + interp_per_complex.json")
    print("  - sequence_attention_heatmaps.png")
    if args.run_length_analysis:
        print("  - length_analysis.png")
    if args.run_attention_comparison:
        print("  - attention_comparison.png + attn_results.json")


if __name__ == "__main__":
    main()
