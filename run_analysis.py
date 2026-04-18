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
                       val_codes, val_targets_std, val_morgan,
                       test_codes, test_targets, test_morgan,
                       t_mean, t_std, train_morgan):
    """Per-layer probes for all 3 ESM sizes."""

    print("\n" + "=" * 80)
    print("STEP 2: LAYER-LEVEL ANALYSIS")
    print("=" * 80)

    layer_results = {}

    for label in ESM_CONFIGS:
        cfg = ESM_CONFIGS[label]
        num_layers = cfg["layers"] + 1

        print(f"\n--- ESM-2 {label} ({num_layers} layers) ---")

        # Load all-layer embeddings
        # We already have these cached from ablation; reuse same cache names
        tr_alllayer = extract_esm2_embeddings(
            [], [], f"jglaser100k", label, all_layers=True)
        va_alllayer = extract_esm2_embeddings(
            [], [], f"pdbbind_val", label, all_layers=True)
        te_alllayer = extract_esm2_embeddings(
            [], [], f"pdbbind_test", label, all_layers=True)

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
                lg = train_morgan[train_ln[i]]
                X_tr.append(np.concatenate([p, lg]))
            X_tr = np.array(X_tr)

            X_te = []
            for code in test_codes:
                p = te_l[code]
                if isinstance(p, torch.Tensor): p = p.numpy()
                lg = test_morgan[code]
                X_te.append(np.concatenate([p, lg]))
            X_te = np.array(X_te)

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X_tr, train_targets_std[tr_idx])
            y_pred = ridge.predict(X_te) * t_std + t_mean
            ci = concordance_index(test_targets, y_pred)
            layer_cis.append(ci)

        layer_results[label] = layer_cis
        peak = np.argmax(layer_cis)
        print(f"  Peak layer: L{peak} (CI={layer_cis[peak]:.4f})")
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
    ax.set_title("Per-Layer Binding Affinity Signal", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("layer_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved layer_analysis.png")

    with open("layer_results.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in layer_results.items()}, f, indent=2)
    print("Saved layer_results.json")

    return layer_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: RESIDUE-LEVEL INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_interpretability(model, test_codes, test_seqs, te_res, te_morgan):
    """Pocket overlap + gradient attribution on CASF-2016 test set."""

    print("\n" + "=" * 80)
    print("STEP 3: RESIDUE-LEVEL INTERPRETABILITY")
    print("=" * 80)

    enrichments, aurocs, grad_aurocs, attn_grad_corrs = [], [], [], []
    all_attn_pocket, all_attn_nonpocket = [], []

    for i, code in enumerate(test_codes):
        _, pocket_mask = get_pocket_mask(code)
        if pocket_mask is None or code not in te_res:
            continue

        residue_embs = te_res[code]
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
        lig = torch.FloatTensor(te_morgan[code]).unsqueeze(0).to(DEVICE)
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

        k = n_pocket
        top_k = np.argsort(attn_w)[-k:]
        prec = pocket_mask[top_k].sum() / k
        expected = n_pocket / len(pocket_mask)
        enrichments.append(prec / expected if expected > 0 else 0)

        try: aurocs.append(roc_auc_score(pocket_mask.astype(int), attn_w))
        except: aurocs.append(0.5)
        try: grad_aurocs.append(roc_auc_score(pocket_mask.astype(int), grad_w))
        except: grad_aurocs.append(0.5)

        corr, _ = stats.spearmanr(attn_w, grad_w)
        attn_grad_corrs.append(corr)

    n = len(enrichments)
    print(f"\nAnalyzed {n} complexes with pocket annotations")
    print(f"  Enrichment: {np.mean(enrichments):.2f}x +/- {np.std(enrichments):.2f}")
    print(f"  Attn AUC-ROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"  Grad AUC-ROC: {np.mean(grad_aurocs):.4f} +/- {np.std(grad_aurocs):.4f}")
    print(f"  Attn-Grad Spearman: {np.mean(attn_grad_corrs):.4f}")

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

    plt.suptitle(f"Interpretability: {conclusion}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("interpretability.png", dpi=150, bbox_inches="tight")
    print("Saved interpretability.png")

    results = {"enrichment": np.mean(enrichments), "auroc": np.mean(aurocs),
               "grad_auroc": np.mean(grad_aurocs),
               "attn_grad_corr": np.mean(attn_grad_corrs),
               "pocket_ratio": pocket_ratio, "p_value": p_val,
               "conclusion": conclusion}
    with open("interp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved interp_results.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: POOLING + LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_length_analysis(mean_model, attn_model, test_codes, test_seqs,
                        test_targets, te_mean, te_res, te_morgan,
                        t_mean, t_std):
    """Compare mean vs attention pooling performance by protein length."""

    print("\n" + "=" * 80)
    print("STEP 4: POOLING + LENGTH ANALYSIS")
    print("=" * 80)

    # Predict with mean-pooled model
    te_concat = ConcatDS(test_codes, test_codes,
                         (test_targets - t_mean) / t_std, te_mean, te_morgan)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(te_concat, 256):
            preds.append(mean_model(x.to(DEVICE)).cpu())
    y_pred_mean = torch.cat(preds).numpy() * t_std + t_mean

    # Predict with attention model
    te_res_ds = ResidueDS(test_codes, test_codes,
                          (test_targets - t_mean) / t_std, te_res, te_morgan)
    preds = []
    with torch.no_grad():
        for r, l, m, _ in DataLoader(te_res_ds, 256, collate_fn=residue_collate_fn):
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
                             val_codes, val_targets_std, val_morgan,
                             test_codes, test_targets, test_morgan,
                             tr_res, va_res, te_res,
                             tr_layer, va_layer, te_layer,
                             train_morgan, t_mean, t_std):
    """4 attention methods × 3 settings."""

    print("\n" + "=" * 80)
    print("4×3 ATTENTION ARCHITECTURE COMPARISON")
    print("=" * 80)

    cfg = ESM_CONFIGS["650M"]
    embed_dim, num_layers = cfg["dim"], cfg["layers"] + 1
    ligand_dim = 2048
    BS = 512
    all_results = []

    # ── Setting A: MLP + Attention (4 variants) ──
    print("\n--- Setting A: MLP + Attention ---")
    tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                      [train_ln[i] for i in tr_idx],
                      train_targets_std[tr_idx], tr_res, train_morgan)
    va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, val_morgan)
    te_ds = ResidueDS(test_codes, test_codes,
                      (test_targets - t_mean) / t_std, te_res, test_morgan)

    for aname, acls in ATTN_METHODS.items():
        name = f"A: MLP+{aname}"
        print(f"\n  {name}")
        model = train_loop(
            AttnMLP(acls, embed_dim, ligand_dim),
            DataLoader(tr_ds, BS, shuffle=True, collate_fn=residue_collate_fn),
            DataLoader(va_ds, BS, collate_fn=residue_collate_fn),
            fwd_residue, patience=20)
        preds = []
        with torch.no_grad():
            for r, l, m, _ in DataLoader(te_ds, 256, collate_fn=residue_collate_fn):
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
                     train_targets_std[tr_idx], tr_layer, train_morgan, num_layers)
    va_lds = LayerDS(val_codes, val_codes, val_targets_std,
                     va_layer, val_morgan, num_layers)
    te_lds = LayerDS(test_codes, test_codes,
                     (test_targets - t_mean) / t_std,
                     te_layer, test_morgan, num_layers)

    name = "B: MLP+LayerW"
    print(f"\n  {name}")
    model = train_loop(
        LayerWMLP(num_layers, embed_dim, ligand_dim),
        DataLoader(tr_lds, BS, shuffle=True), DataLoader(va_lds, BS),
        fwd_layer, patience=20)
    preds = []
    with torch.no_grad():
        for ly, lg, _ in DataLoader(te_lds, 256):
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
                            tr_layer, tr_res, train_morgan, num_layers)
    va_lrs = LayerResidueDS(val_codes, val_codes, val_targets_std,
                            va_layer, va_res, val_morgan, num_layers)
    te_lrs = LayerResidueDS(test_codes, test_codes,
                            (test_targets - t_mean) / t_std,
                            te_layer, te_res, test_morgan, num_layers)

    for aname, acls in ATTN_METHODS.items():
        name = f"C: MLP+LayerW+{aname}"
        print(f"\n  {name}")
        model = train_loop(
            LayerWAttnMLP(acls, num_layers, embed_dim, ligand_dim),
            DataLoader(tr_lrs, BS, shuffle=True, collate_fn=layer_residue_collate_fn),
            DataLoader(va_lrs, BS, collate_fn=layer_residue_collate_fn),
            fwd_layer_residue, patience=20)
        preds = []
        with torch.no_grad():
            for ly, rs, lg, mk, _ in DataLoader(
                    te_lrs, 256, collate_fn=layer_residue_collate_fn):
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
    ax.set_title("4 Attention Methods × 3 Settings", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"Setting {s}")
                       for s, c in colors_s.items()], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("attention_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved attention_comparison.png")

    with open("attn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved attn_results.json")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("ANALYSIS: Layers, Residues, Length, Attention Architectures")
    print("Train: jglaser 10k | Val: PDBbind refined-core | Test: CASF-2016")
    print("=" * 80)

    # ── Load data ──
    train_seqs, train_smiles, train_targets, train_pn, train_ln = load_jglaser(10000)
    t_mean, t_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - t_mean) / t_std

    (val_codes, val_seqs, val_smiles, val_targets,
     test_codes, test_seqs, test_smiles, test_targets) = load_pdbbind_val_test()
    val_targets_std = (val_targets - t_mean) / t_std

    n = len(train_targets)
    perm = np.random.permutation(n)
    tr_idx = perm[:int(0.9 * n)]

    # ── Extract embeddings (650M for steps 3-4 and attention comparison) ──
    train_unames = list(set(train_pn))
    train_useqs = list(set(zip(train_pn, train_seqs)))
    train_useqs_only = [s for _, s in train_useqs]
    train_unames_only = [n for n, _ in train_useqs]

    train_smi_pairs = list(set(zip(train_ln, train_smiles)))
    train_ulnames = [n for n, _ in train_smi_pairs]
    train_usmiles = [s for _, s in train_smi_pairs]

    train_morgan = compute_morgan(train_usmiles, train_ulnames)
    val_morgan = compute_morgan(val_smiles, val_codes)
    test_morgan = compute_morgan(test_smiles, test_codes)

    # 650M embeddings
    tr_res = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                      "jglaser100k", "650M", per_residue=True)
    va_res = extract_esm2_embeddings(val_seqs, val_codes,
                                      "pdbbind_val", "650M", per_residue=True)
    te_res = extract_esm2_embeddings(test_seqs, test_codes,
                                      "pdbbind_test", "650M", per_residue=True)

    tr_mean = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                       "jglaser100k", "650M")
    te_mean = extract_esm2_embeddings(test_seqs, test_codes,
                                       "pdbbind_test", "650M")

    tr_layer = extract_esm2_embeddings(train_useqs_only, train_unames_only,
                                        "jglaser100k", "650M", all_layers=True)
    va_layer = extract_esm2_embeddings(val_seqs, val_codes,
                                        "pdbbind_val", "650M", all_layers=True)
    te_layer = extract_esm2_embeddings(test_seqs, test_codes,
                                        "pdbbind_test", "650M", all_layers=True)

    # ── Step 2: Layer analysis ──
    run_layer_analysis(train_pn, train_ln, train_targets_std, tr_idx,
                       val_codes, val_targets_std, val_morgan,
                       test_codes, test_targets, test_morgan,
                       t_mean, t_std, train_morgan)

    # ── Step 3: Interpretability (train attention model first) ──
    print("\nTraining attention model for interpretability...")
    cfg = ESM_CONFIGS["650M"]
    tr_ds = ResidueDS([train_pn[i] for i in tr_idx],
                      [train_ln[i] for i in tr_idx],
                      train_targets_std[tr_idx], tr_res, train_morgan)
    va_ds = ResidueDS(val_codes, val_codes, val_targets_std, va_res, val_morgan)
    attn_model = train_loop(
        AttnMLP(ProteinOnlyAttn, cfg["dim"], 2048),
        DataLoader(tr_ds, 512, shuffle=True, collate_fn=residue_collate_fn),
        DataLoader(va_ds, 512, collate_fn=residue_collate_fn),
        fwd_residue, patience=20)

    run_interpretability(attn_model, test_codes, test_seqs, te_res, test_morgan)

    # ── Step 4: Length analysis (need mean model too) ──
    va_mean = extract_esm2_embeddings(val_seqs, val_codes, "pdbbind_val", "650M")
    va_concat = ConcatDS(val_codes, val_codes, val_targets_std, va_mean, val_morgan)
    tr_concat = ConcatDS([train_pn[i] for i in tr_idx],
                         [train_ln[i] for i in tr_idx],
                         train_targets_std[tr_idx], tr_mean, train_morgan)
    mean_model = train_loop(
        MLPProbe(cfg["dim"] + 2048),
        DataLoader(tr_concat, 512, shuffle=True, num_workers=4),
        DataLoader(va_concat, 512, num_workers=4),
        fwd_concat, patience=20)

    run_length_analysis(mean_model, attn_model, test_codes, test_seqs,
                        test_targets, te_mean, te_res, test_morgan, t_mean, t_std)
    del mean_model, attn_model; torch.cuda.empty_cache()

    # ── 4×3 Attention comparison ──
    run_attention_comparison(
        train_pn, train_ln, train_targets_std, tr_idx,
        val_codes, val_targets_std, val_morgan,
        test_codes, test_targets, test_morgan,
        tr_res, va_res, te_res, tr_layer, va_layer, te_layer,
        train_morgan, t_mean, t_std)

    print("\n" + "=" * 80)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 80)
    print("Outputs: layer_analysis.png, interpretability.png, "
          "length_analysis.png, attention_comparison.png")


if __name__ == "__main__":
    main()
