"""Visualize ablation study results from results.json."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

with open("results.json") as f:
    data = json.load(f)

# ── Organize data ─────────────────────────────────────────────────────────────
sizes = ["8M", "35M", "650M"]
pools = ["mean", "max", "attn"]
ligands = ["Morgan", "ChemBERTa"]
models = ["Ridge", "MLP", "LayerW+MLP"]

def lookup(model, size, pool, fp, metric="CI"):
    for d in data:
        if (d["Model"] == model and d["Size"] == size
                and d["Pool"] == pool and d["Fingerprint"] == fp):
            return d[metric]
    return None

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Ablation Study: Binding Affinity Prediction on Davis Dataset",
             fontsize=16, fontweight="bold", y=0.98)

colors = {"Morgan": "#2196F3", "ChemBERTa": "#FF9800"}
size_colors = {"8M": "#66BB6A", "35M": "#FFA726", "650M": "#EF5350"}
pool_colors = {"mean": "#42A5F5", "max": "#AB47BC", "attn": "#26A69A"}

# ── Plot 1: Pooling effect (MLP only, CI) ─────────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(sizes))
width = 0.12
offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

i = 0
for pool in pools:
    for fp in ligands:
        vals = []
        for size in sizes:
            v = lookup("MLP", size, pool, fp)
            vals.append(v if v is not None else 0)
        label = f"{pool} + {fp}"
        color = pool_colors[pool]
        alpha = 1.0 if fp == "Morgan" else 0.55
        bars = ax1.bar(x + offsets[i] * width, vals, width, label=label,
                       color=color, alpha=alpha, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
        i += 1

ax1.set_xlabel("ESM-2 Model Size")
ax1.set_ylabel("Concordance Index (CI)")
ax1.set_title("Effect of Pooling Strategy (MLP Probe)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.set_ylim(0.80, 0.89)
ax1.legend(fontsize=7, ncol=2, loc="upper left")
ax1.axhline(y=0.85, color="gray", linestyle="--", alpha=0.3)
ax1.grid(axis="y", alpha=0.3)

# ── Plot 2: Scale effect across probe types ───────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)

# Best config per probe type for each size (Morgan FP)
probe_configs = {
    "Ridge (mean)":     [lookup("Ridge", s, "mean", "Morgan") for s in sizes],
    "MLP (mean)":       [lookup("MLP", s, "mean", "Morgan") for s in sizes],
    "MLP (attn)":       [lookup("MLP", s, "attn", "Morgan") for s in sizes],
    "LayerW+MLP":       [lookup("LayerW+MLP", s, "mean", "Morgan") for s in sizes],
}

markers = {"Ridge (mean)": "s", "MLP (mean)": "o", "MLP (attn)": "D", "LayerW+MLP": "^"}
line_colors = ["#78909C", "#42A5F5", "#26A69A", "#EF5350"]

for (label, vals), color in zip(probe_configs.items(), line_colors):
    ax2.plot(sizes, vals, marker=markers[label], label=label, color=color,
             linewidth=2, markersize=8)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:.4f}", (sizes[i], v), textcoords="offset points",
                     xytext=(8, -3), fontsize=7)

ax2.set_xlabel("ESM-2 Model Size")
ax2.set_ylabel("Concordance Index (CI)")
ax2.set_title("Effect of Scale Across Probe Types (Morgan FP)", fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_ylim(0.78, 0.89)
ax2.grid(alpha=0.3)

# ── Plot 3: Ligand representation comparison ─────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)

configs = []
morgan_vals = []
chemberta_vals = []
for model in ["MLP", "MLP", "MLP", "LayerW+MLP"]:
    pool = "attn" if model == "MLP" and len(configs) >= 3 else "mean"
    if model == "MLP" and len(configs) < 3:
        pool = ["mean", "max", "attn"][len(configs)]
    for size in ["650M"]:
        m = lookup(model, size, pool, "Morgan")
        c = lookup(model, size, pool, "ChemBERTa")
        if m is not None and c is not None:
            configs.append(f"{model}\n{pool} | {size}")
            morgan_vals.append(m)
            chemberta_vals.append(c)

# Redo more cleanly
configs, morgan_vals, chemberta_vals = [], [], []
combos = [
    ("MLP", "650M", "mean"), ("MLP", "650M", "max"), ("MLP", "650M", "attn"),
    ("LayerW+MLP", "650M", "mean"),
    ("MLP", "8M", "mean"), ("MLP", "8M", "attn"),
]
for model, size, pool in combos:
    m = lookup(model, size, pool, "Morgan")
    c = lookup(model, size, pool, "ChemBERTa")
    if m is not None and c is not None:
        configs.append(f"{model}\n{pool} | {size}")
        morgan_vals.append(m)
        chemberta_vals.append(c)

x = np.arange(len(configs))
width = 0.35
bars1 = ax3.bar(x - width/2, morgan_vals, width, label="Morgan FP",
                color=colors["Morgan"], alpha=0.85)
bars2 = ax3.bar(x + width/2, chemberta_vals, width, label="ChemBERTa",
                color=colors["ChemBERTa"], alpha=0.85)

for bars in [bars1, bars2]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

ax3.set_ylabel("Concordance Index (CI)")
ax3.set_title("Morgan FP vs ChemBERTa Ligand Embeddings", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(configs, fontsize=7)
ax3.set_ylim(0.80, 0.89)
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)

# ── Plot 4: Bottleneck analysis (CI spread per factor) ───────────────────────
ax4 = fig.add_subplot(2, 2, 4)

factor_spreads = {}
factor_details = {}

# Probe effect: compare Ridge vs MLP vs LayerW+MLP (mean pool, Morgan, 650M)
probe_vals = {
    "Ridge":      lookup("Ridge", "650M", "mean", "Morgan"),
    "MLP (mean)": lookup("MLP", "650M", "mean", "Morgan"),
    "MLP (attn)": lookup("MLP", "650M", "attn", "Morgan"),
    "LayerW+MLP": lookup("LayerW+MLP", "650M", "mean", "Morgan"),
}
vals = list(probe_vals.values())
factor_spreads["Probe\nArchitecture"] = max(vals) - min(vals)
factor_details["Probe\nArchitecture"] = probe_vals

# Pooling effect: mean vs max vs attn (MLP, Morgan, 650M)
pool_vals = {p: lookup("MLP", "650M", p, "Morgan") for p in pools}
vals = [v for v in pool_vals.values() if v is not None]
factor_spreads["Pooling\nStrategy"] = max(vals) - min(vals)
factor_details["Pooling\nStrategy"] = pool_vals

# Scale effect: 8M vs 35M vs 650M (MLP, attn, Morgan)
scale_vals = {s: lookup("MLP", s, "attn", "Morgan") for s in sizes}
vals = list(scale_vals.values())
factor_spreads["Model\nScale"] = max(vals) - min(vals)
factor_details["Model\nScale"] = scale_vals

# Ligand effect: Morgan vs ChemBERTa (MLP, attn, 650M)
lig_vals = {fp: lookup("MLP", "650M", "attn", fp) for fp in ligands}
vals = list(lig_vals.values())
factor_spreads["Ligand\nRepr"] = max(vals) - min(vals)
factor_details["Ligand\nRepr"] = lig_vals

factors = list(factor_spreads.keys())
spreads = list(factor_spreads.values())
bar_colors = ["#EF5350", "#26A69A", "#FFA726", "#42A5F5"]

bars = ax4.bar(factors, spreads, color=bar_colors, alpha=0.85, edgecolor="white",
               linewidth=1.5, width=0.6)
for bar, spread in zip(bars, spreads):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"\u0394CI={spread:.4f}", ha="center", va="bottom", fontsize=9,
             fontweight="bold")

# Add detail annotations
for i, (factor, details) in enumerate(factor_details.items()):
    detail_str = "\n".join(f"{k}: {v:.4f}" for k, v in details.items() if v is not None)
    ax4.annotate(detail_str, (i, spreads[i]/2), ha="center", va="center",
                 fontsize=6, color="white", fontweight="bold")

ax4.set_ylabel("CI Spread (max - min)")
ax4.set_title("Bottleneck Analysis: Which Component Matters Most?",
              fontweight="bold")
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ablation_results.png", dpi=150, bbox_inches="tight")
print("Saved to ablation_results.png")


# ── Print numerical summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print("\n1. POOLING (information loss from mean pooling):")
print(f"   Mean pooling (650M, Morgan):  CI = {lookup('MLP', '650M', 'mean', 'Morgan'):.4f}")
print(f"   Max pooling  (650M, Morgan):  CI = {lookup('MLP', '650M', 'max', 'Morgan'):.4f}")
print(f"   Attn pooling (650M, Morgan):  CI = {lookup('MLP', '650M', 'attn', 'Morgan'):.4f}")
mean_ci = lookup("MLP", "650M", "mean", "Morgan")
attn_ci = lookup("MLP", "650M", "attn", "Morgan")
print(f"   -> Attention recovers +{attn_ci - mean_ci:.4f} CI over mean pooling")
print(f"   -> Max pooling HURTS: loses {mean_ci - lookup('MLP', '650M', 'max', 'Morgan'):.4f} CI vs mean")

print("\n2. SCALE (how model size affects performance):")
for s in sizes:
    ci = lookup("MLP", s, "attn", "Morgan")
    print(f"   ESM-2 {s:>4s} (attn, Morgan): CI = {ci:.4f}")
print(f"   -> 650M vs 8M delta: +{lookup('MLP', '650M', 'attn', 'Morgan') - lookup('MLP', '8M', 'attn', 'Morgan'):.4f}")
print(f"   -> Scale matters MORE with nonlinear probes (Ridge spread: ~0.003, MLP attn spread: ~0.022)")

print("\n3. LIGAND REPRESENTATION:")
print(f"   Morgan FP  (MLP, attn, 650M): CI = {lookup('MLP', '650M', 'attn', 'Morgan'):.4f}")
print(f"   ChemBERTa  (MLP, attn, 650M): CI = {lookup('MLP', '650M', 'attn', 'ChemBERTa'):.4f}")
print(f"   -> ChemBERTa slightly WORSE for best configs (-{lookup('MLP', '650M', 'attn', 'Morgan') - lookup('MLP', '650M', 'attn', 'ChemBERTa'):.4f})")
print(f"   -> ChemBERTa BETTER for smaller models: 8M mean: {lookup('MLP', '8M', 'mean', 'ChemBERTa'):.4f} vs {lookup('MLP', '8M', 'mean', 'Morgan'):.4f}")
print(f"   -> LayerW+MLP strongly prefers Morgan: 650M: {lookup('LayerW+MLP', '650M', 'mean', 'Morgan'):.4f} vs {lookup('LayerW+MLP', '650M', 'mean', 'ChemBERTa'):.4f}")

print("\n4. BOTTLENECK RANKING (by CI spread at best config):")
sorted_factors = sorted(factor_spreads.items(), key=lambda x: -x[1])
for rank, (factor, spread) in enumerate(sorted_factors, 1):
    print(f"   {rank}. {factor.replace(chr(10), ' '):<25s}: spread = {spread:.4f}")

print("\n5. BEST OVERALL CONFIG:")
best = max(data, key=lambda d: d["CI"])
print(f"   {best['Model']} | {best['Size']} | {best['Pool']} | {best['Fingerprint']}")
print(f"   CI={best['CI']:.4f} | MSE={best['MSE']:.4f} | R²={best['R^2']:.4f} | r={best['r']:.4f}")
