"""
Aggregate the per-ligand layer + interpretability JSONs (Morgan, ChemBERTa,
MolFormer) into combined figures for the paper.

Inputs (already on disk):
  layer_results_{Morgan,ChemBERTa,MolFormer}.json
  interp_results_{Morgan,ChemBERTa,MolFormer}.json
  interp_per_complex_{Morgan,ChemBERTa,MolFormer}.json

Outputs:
  layer_analysis_all_ligands.png       # 3 subplots (ESM size) x 3 lines (ligand)
  interpretability_all_ligands.png     # bar chart: enrichment, AUC, P@10 per ligand
  faithfulness_all_ligands.png         # Spearman(attn, grad) histograms overlaid
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import RESULTS_DIR, FIGURES_DIR

LIGANDS = ["Morgan", "ChemBERTa", "MolFormer"]
COLORS = {"Morgan": "#42A5F5", "ChemBERTa": "#EF5350", "MolFormer": "#66BB6A"}
SIZES = ["8M", "35M", "650M"]


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


layer = {l: load(f"layer_results_{l}.json") for l in LIGANDS}
interp = {l: load(f"interp_results_{l}.json") for l in LIGANDS}
per_complex = {l: load(f"interp_per_complex_{l}.json") for l in LIGANDS}


# ── Figure 1: CI vs layer, one panel per ESM size, one line per ligand ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
for ax, size in zip(axes[0], SIZES):
    for lig in LIGANDS:
        cis = layer[lig][size]
        ax.plot(range(len(cis)), cis, marker="o", markersize=3.5,
                linewidth=1.8, color=COLORS[lig], label=lig)
        peak = int(np.argmax(cis))
        ax.annotate(f"L{peak}", (peak, cis[peak]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=COLORS[lig])
    ax.set_title(f"ESM-2 {size}", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CI (Ridge probe)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "layer_analysis_all_ligands.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved layer_analysis_all_ligands.png")


# ── Figure 2: interpretability bar chart across ligands ──
metrics = [
    ("enrichment",   "Pocket Enrichment",          1.0),
    ("auroc",        "Attention AUC-ROC",          0.5),
    ("grad_auroc",   "Gradient AUC-ROC",           0.5),
    ("attn_grad_corr", "Attn↔Grad Spearman ρ",     0.0),
]

fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
x = np.arange(len(LIGANDS))
for ax, (key, title, ref) in zip(axes, metrics):
    vals = [interp[l][key] for l in LIGANDS]
    bars = ax.bar(x, vals, color=[COLORS[l] for l in LIGANDS],
                  alpha=0.85, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.2f}" if abs(v) >= 0.1 else f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(ref, color="red", linestyle="--", alpha=0.6,
               label=f"chance ({ref})")
    ax.set_xticks(x)
    ax.set_xticklabels(LIGANDS)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
plt.suptitle("Residue-level interpretability across ligand encoders "
             "(650M ESM-2, 280 PDBbind Test2016 complexes)",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIGURES_DIR, "interpretability_all_ligands.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved interpretability_all_ligands.png")


# ── Figure 3: per-complex Spearman(attn, grad) histograms overlaid ──
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-0.5, 0.7, 40)
for lig in LIGANDS:
    rhos = [c["attn_grad_spearman"] for c in per_complex[lig]
            if c["attn_grad_spearman"] is not None
            and not np.isnan(c["attn_grad_spearman"])]
    ax.hist(rhos, bins=bins, alpha=0.55, color=COLORS[lig],
            label=f"{lig} (mean={np.mean(rhos):.3f})", density=True)
    ax.axvline(np.mean(rhos), color=COLORS[lig], linestyle="--",
               linewidth=1.5, alpha=0.9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman ρ (attention vs gradient norm), per complex")
ax.set_ylabel("Density")
ax.set_title("Attention faithfulness by ligand encoder", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "faithfulness_all_ligands.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved faithfulness_all_ligands.png")


# ── Summary tables to stdout (LaTeX-pasteable) ──
print("\n--- Layer summary (peak layer / peak CI / std across layers) ---")
print(f"{'Ligand':<10s} | " + " | ".join(f"{s:^22s}" for s in SIZES))
for lig in LIGANDS:
    cells = []
    for s in SIZES:
        cis = layer[lig][s]
        peak = int(np.argmax(cis))
        cells.append(f"L{peak} / {cis[peak]:.4f} / {np.std(cis):.4f}")
    print(f"{lig:<10s} | " + " | ".join(f"{c:^22s}" for c in cells))

print("\n--- Interpretability summary ---")
hdr = ["enrichment", "auroc", "grad_auroc", "attn_grad_corr",
       "precision_at_k.10", "precision_at_k.25", "precision_at_k.50"]
print(f"{'Ligand':<10s} | " + " | ".join(f"{h:>14s}" for h in hdr))
for lig in LIGANDS:
    r = interp[lig]
    pk = r.get("precision_at_k", {})
    row = [r["enrichment"], r["auroc"], r["grad_auroc"], r["attn_grad_corr"],
           pk.get("10", 0), pk.get("25", 0), pk.get("50", 0)]
    print(f"{lig:<10s} | " + " | ".join(f"{v:>14.4f}" for v in row))
