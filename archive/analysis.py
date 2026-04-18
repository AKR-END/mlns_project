"""
Post-ablation analysis on Davis dataset:
  1. Error analysis by protein sequence length (does pooling loss scale with length?)
  2. Attention weight visualization (what residues does attention focus on?)

Uses cached embeddings from ablation_study.py. Davis dataset only.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/ssd_scratch/akr"
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
os.environ["TORCH_HOME"] = CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import (
    download_davis, load_davis, compute_ligand_fingerprints, concordance_index,
)
from ablation_study import (
    ESM_CONFIGS, extract_residue_embeddings,
    pool_residues, build_feature_matrix,
    train_mlp_probe, predict_mlp,
    train_attention_mlp, predict_attention,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ERROR ANALYSIS BY PROTEIN LENGTH
# ═══════════════════════════════════════════════════════════════════════════════

def error_analysis_by_length(df, train_idx, val_idx, test_idx,
                             residue_dict, morgan_dict, y_test):
    cfg = ESM_CONFIGS["650M"]

    print("\n" + "=" * 70)
    print("1. ERROR ANALYSIS BY PROTEIN SEQUENCE LENGTH")
    print("=" * 70)

    # Train mean-pooled MLP
    print("\nTraining mean-pooled MLP (650M, Morgan)...")
    prot_pooled = pool_residues(residue_dict, "mean")
    X = build_feature_matrix(df, prot_pooled, morgan_dict)
    y_all = df["pKd"].values
    mean_model = train_mlp_probe(X[train_idx], y_all[train_idx],
                                  X[val_idx], y_all[val_idx])
    y_pred_mean = predict_mlp(mean_model, X[test_idx])

    # Train attention-pooled MLP
    print("Training attention-pooled MLP (650M, Morgan)...")
    attn_model = train_attention_mlp(
        df, train_idx, val_idx, residue_dict, morgan_dict,
        embed_dim=cfg["dim"], ligand_dim=2048)
    y_pred_attn = predict_attention(attn_model, df, test_idx,
                                    residue_dict, morgan_dict)

    errors_mean = np.abs(y_test - y_pred_mean)
    errors_attn = np.abs(y_test - y_pred_attn)
    seq_lengths = df.iloc[test_idx]["sequence"].str.len().values

    bins = [0, 300, 500, 700, 900, 1100]
    bin_labels = ["<300", "300-500", "500-700", "700-900", "900+"]
    bucket_idx = np.clip(np.digitize(seq_lengths, bins) - 1, 0, len(bin_labels) - 1)

    print(f"\n{'Bucket':<12s} | {'N':>5s} | {'MAE mean':>9s} | {'MAE attn':>9s} | "
          f"{'CI mean':>8s} | {'CI attn':>8s} | {'Delta CI':>8s}")
    print("-" * 75)

    bucket_stats = []
    for i, label in enumerate(bin_labels):
        mask = bucket_idx == i
        n = mask.sum()
        if n < 10:
            continue
        mae_m = errors_mean[mask].mean()
        mae_a = errors_attn[mask].mean()
        ci_m = concordance_index(y_test[mask], y_pred_mean[mask])
        ci_a = concordance_index(y_test[mask], y_pred_attn[mask])
        delta = ci_a - ci_m
        print(f"  {label:<10s} | {n:>5d} | {mae_m:>9.4f} | {mae_a:>9.4f} | "
              f"{ci_m:>8.4f} | {ci_a:>8.4f} | {delta:>+8.4f}")
        bucket_stats.append({"label": label, "n": n, "mae_mean": mae_m,
                              "mae_attn": mae_a, "ci_mean": ci_m,
                              "ci_attn": ci_a, "delta_ci": delta})

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [b["label"] for b in bucket_stats]
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x - 0.2, [b["mae_mean"] for b in bucket_stats], 0.35,
           label="Mean pool", color="#42A5F5", alpha=0.85)
    ax.bar(x + 0.2, [b["mae_attn"] for b in bucket_stats], 0.35,
           label="Attn pool", color="#26A69A", alpha=0.85)
    ax.set_xlabel("Protein Sequence Length")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Prediction Error by Protein Length", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 0.2, [b["ci_mean"] for b in bucket_stats], 0.35,
           label="Mean pool", color="#42A5F5", alpha=0.85)
    ax.bar(x + 0.2, [b["ci_attn"] for b in bucket_stats], 0.35,
           label="Attn pool", color="#26A69A", alpha=0.85)
    ax.set_xlabel("Protein Sequence Length")
    ax.set_ylabel("Concordance Index (CI)")
    ax.set_title("CI by Protein Length", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("length_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved length_analysis.png")

    del mean_model, attn_model
    torch.cuda.empty_cache()
    return bucket_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ATTENTION WEIGHT VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_attention_weights(df, train_idx, val_idx, test_idx,
                                residue_dict, morgan_dict):
    cfg = ESM_CONFIGS["650M"]

    print("\n" + "=" * 70)
    print("2. ATTENTION WEIGHT VISUALIZATION")
    print("=" * 70)

    print("Training attention-pooled MLP (650M, Morgan)...")
    model = train_attention_mlp(
        df, train_idx, val_idx, residue_dict, morgan_dict,
        embed_dim=cfg["dim"], ligand_dim=2048)

    # Pick 5 proteins of varying length from test set
    test_prots = df.iloc[test_idx][["protein_name", "sequence"]].drop_duplicates()
    test_prots = test_prots.copy()
    test_prots["length"] = test_prots["sequence"].str.len()
    test_prots = test_prots.sort_values("length")
    n = len(test_prots)
    samples = test_prots.iloc[[0, n//4, n//2, 3*n//4, n-1]]

    fig, axes = plt.subplots(len(samples), 1, figsize=(16, 3 * len(samples)))

    for i, (_, prow) in enumerate(samples.iterrows()):
        pname = prow["protein_name"]
        seq = prow["sequence"][:1022]
        seq_len = len(seq)

        residues = residue_dict[pname].unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, seq_len, dtype=torch.bool, device=DEVICE)

        model.eval()
        with torch.no_grad():
            scores = model.attn(residues).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1)[0].cpu().numpy()

        ax = axes[i]
        ax.bar(range(seq_len), weights, width=1.0, color="#26A69A", alpha=0.7)
        top10 = np.argsort(weights)[-10:][::-1]
        for idx in top10:
            ax.bar(idx, weights[idx], width=1.0, color="#EF5350", alpha=0.9)

        ax.set_ylabel("Attn Weight", fontsize=8)
        ax.set_title(f"{pname} (len={seq_len})", fontsize=10, fontweight="bold")
        ax.set_xlim(0, seq_len)

        entropy = -np.sum(weights * np.log(weights + 1e-10))
        uniformity = entropy / np.log(seq_len)
        ax.text(0.02, 0.85, f"Entropy ratio: {uniformity:.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.5))

    axes[-1].set_xlabel("Residue Position")
    plt.suptitle("Attention Weights Over Protein Residues (ESM-2 650M)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=150, bbox_inches="tight")
    print("Saved attention_weights.png")

    del model
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("POST-ABLATION ANALYSIS")
    print("=" * 70)

    download_davis()
    df = load_davis()

    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=42)
    y_test = df.iloc[test_idx]["pKd"].values

    morgan_dict = compute_ligand_fingerprints(df)
    residue_dict = extract_residue_embeddings(df, "650M")

    length_stats = error_analysis_by_length(
        df, train_idx, val_idx, test_idx, residue_dict, morgan_dict, y_test)

    visualize_attention_weights(
        df, train_idx, val_idx, test_idx, residue_dict, morgan_dict)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if length_stats:
        long = [b for b in length_stats if "900" in b["label"]]
        short = [b for b in length_stats if "<300" in b["label"]]
        if long and short:
            print(f"  Attn vs Mean CI delta for short (<300): {short[0]['delta_ci']:+.4f}")
            print(f"  Attn vs Mean CI delta for long  (900+): {long[0]['delta_ci']:+.4f}")
    print("  See length_analysis.png and attention_weights.png")


if __name__ == "__main__":
    main()
