"""
Audit train/test leakage in BAPULM's published benchmark CSVs against the
jglaser/binding_affinity training set (first 100k rows, the slice both BAPULM
and we use for training).

Produces, for the paper:
  leakage_summary.json   per-test-set leakage rates + reported BAPULM scores
  leakage_examples.md    concrete (protein, ligand, affinity) rows that appear
                         identically in both train and test
  leakage_audit.png      bar-chart figure: leakage rate vs reported BAPULM r
"""

import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import RESULTS_DIR, FIGURES_DIR, DATA_EXTRA_DIR


# BAPULM-released test-set CSVs (cloned from github.com/radh55sh/BAPULM)
BAPULM_DATA = DATA_EXTRA_DIR
TEST_SETS = {
    "benchmark1k2101": {
        "csv": os.path.join(BAPULM_DATA, "benchmark1k2101.csv"),
        "bapulm_r": 0.925,
        "bapulm_rmse": 0.745,
    },
    "Test2016_290": {
        "csv": os.path.join(BAPULM_DATA, "Test2016_290.csv"),
        "bapulm_r": 0.914,
        "bapulm_rmse": 0.898,
    },
    "CSAR-HiQ_36": {
        "csv": os.path.join(BAPULM_DATA, "CSAR-HiQ_36.csv"),
        "bapulm_r": 0.813,
        "bapulm_rmse": 1.328,
    },
}



def load_jglaser_first_n(n=100000):
    parquet = sorted(glob.glob(
        "/ssd_scratch/akr/hf_cache/datasets--jglaser--binding_affinity/"
        "snapshots/*/data/*.parquet"))[0]
    pf = pq.ParquetFile(parquet)
    chunks, taken = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(
            rg, columns=["seq", "smiles_can", "neg_log10_affinity_M"]
        ).to_pandas()
        if taken + len(t) >= n:
            chunks.append(t.iloc[: n - taken]); break
        chunks.append(t); taken += len(t)
    return pd.concat(chunks, ignore_index=True)


def main():
    print("Loading jglaser first 100k...")
    jg = load_jglaser_first_n(100000)
    jg["seq"] = jg["seq"].astype(str)
    jg["smiles_can"] = jg["smiles_can"].astype(str)
    print(f"  jglaser-100k rows: {len(jg)}")

    # Index jglaser by (seq, smiles) for fast lookup of the FIRST training row
    # that matches each leaked test pair. We keep the first match's affinity so
    # the report can show the literal training-row affinity vs. the test-row
    # affinity for the same (protein, ligand).
    jg_lookup = {}
    for _, row in jg.iterrows():
        key = (row["seq"], row["smiles_can"])
        if key not in jg_lookup:
            jg_lookup[key] = float(row["neg_log10_affinity_M"])
    jg_seqs = set(jg["seq"])
    jg_smis = set(jg["smiles_can"])

    summary = {"jglaser_subset": "first 100k", "test_sets": {}}
    examples_md = ["# Concrete leakage examples\n",
                   "Each row below is a (protein, ligand) pair that appears "
                   "**identically** in BAPULM's published test CSV and in the "
                   "first 100k rows of `jglaser/binding_affinity` (BAPULM's "
                   "training data, per their `main.py`/`config.yaml`).",
                   "",
                   "Source: cloned from `github.com/radh55sh/BAPULM` on "
                   "2026-05-03.",
                   ""]

    for name, meta in TEST_SETS.items():
        df = pd.read_csv(meta["csv"])
        df["seq"] = df["seq"].astype(str)
        df["smiles_can"] = df["smiles_can"].astype(str)
        N = len(df)

        leaked_pairs, leaked_seq, leaked_smi = 0, 0, 0
        leaked_pair_examples = []
        for _, row in df.iterrows():
            s, m = row["seq"], row["smiles_can"]
            if (s, m) in jg_lookup:
                leaked_pairs += 1
                if len(leaked_pair_examples) < 5:
                    pdbid = row.get("pdbid", "?")
                    test_aff = float(row["neg_log10_affinity_M"])
                    train_aff = jg_lookup[(s, m)]
                    leaked_pair_examples.append({
                        "pdbid": pdbid,
                        "test_affinity": test_aff,
                        "train_affinity": train_aff,
                        "seq_len": len(s),
                        "smiles": m,
                        "seq_preview": s[:60] + ("..." if len(s) > 60 else ""),
                    })
            if s in jg_seqs:   leaked_seq += 1
            if m in jg_smis:   leaked_smi += 1

        info = {
            "n": N,
            "n_leaked_pairs": leaked_pairs,
            "frac_leaked_pairs": leaked_pairs / N,
            "n_leaked_protein_seq": leaked_seq,
            "frac_leaked_protein_seq": leaked_seq / N,
            "n_leaked_ligand_smi": leaked_smi,
            "frac_leaked_ligand_smi": leaked_smi / N,
            "bapulm_reported_r": meta["bapulm_r"],
            "bapulm_reported_rmse": meta["bapulm_rmse"],
            "csv_path": meta["csv"],
            "examples": leaked_pair_examples,
        }
        summary["test_sets"][name] = info

        print(f"\n{name} (n={N}):")
        print(f"  pairs leaked   : {leaked_pairs:>4d} ({100*leaked_pairs/N:5.1f}%)")
        print(f"  proteins leaked: {leaked_seq:>4d} ({100*leaked_seq/N:5.1f}%)")
        print(f"  ligands leaked : {leaked_smi:>4d} ({100*leaked_smi/N:5.1f}%)")
        print(f"  BAPULM reported r = {meta['bapulm_r']}, "
              f"RMSE = {meta['bapulm_rmse']}")

        examples_md.append(f"## {name}  —  {leaked_pairs}/{N} pairs leaked "
                           f"({100*leaked_pairs/N:.1f}%)")
        examples_md.append("")
        examples_md.append("| pdbid | test affinity | train affinity (jglaser) | seq len | smiles |")
        examples_md.append("|:-----|--------------:|-------------------------:|--------:|:-------|")
        if not leaked_pair_examples:
            examples_md.append("| — | — | — | — | (no exact pair matches) |")
        for ex in leaked_pair_examples:
            examples_md.append(
                f"| {ex['pdbid']} | {ex['test_affinity']:.3f} | "
                f"{ex['train_affinity']:.3f} | {ex['seq_len']} | "
                f"`{ex['smiles']}` |"
            )
        examples_md.append("")

    # ── Save JSON + Markdown ──
    out_json = os.path.join(RESULTS_DIR, "leakage_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved {out_json}")

    out_md = os.path.join(RESULTS_DIR, "leakage_examples.md")
    with open(out_md, "w") as f:
        f.write("\n".join(examples_md))
    print(f"Saved {out_md}")

    # ── Figure: leakage rate vs reported BAPULM r ──
    names = list(TEST_SETS.keys())
    leak_pair = [summary["test_sets"][n]["frac_leaked_pairs"] * 100 for n in names]
    leak_seq  = [summary["test_sets"][n]["frac_leaked_protein_seq"] * 100 for n in names]
    leak_smi  = [summary["test_sets"][n]["frac_leaked_ligand_smi"] * 100 for n in names]
    bapulm_r  = [summary["test_sets"][n]["bapulm_reported_r"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, leak_pair, w, color="#EF5350", label="(protein, ligand) pairs")
    ax.bar(x,       leak_seq,  w, color="#42A5F5", label="proteins")
    ax.bar(x + w,   leak_smi,  w, color="#66BB6A", label="ligands")
    for xi, lp in zip(x, leak_pair):
        ax.text(xi - w, lp + 1, f"{lp:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("% of test set found verbatim in jglaser-100k training")
    ax.set_ylim(0, 110)
    ax.set_title("BAPULM benchmark contamination against its own training set",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Right axis: BAPULM's reported Pearson r
    ax2 = ax.twinx()
    ax2.plot(x, bapulm_r, marker="o", linestyle="--", color="black",
             linewidth=1.5, markersize=8, label="BAPULM reported r")
    for xi, r in zip(x, bapulm_r):
        ax2.text(xi, r + 0.012, f"r={r:.3f}", ha="center", fontsize=8)
    ax2.set_ylabel("BAPULM reported Pearson r")
    ax2.set_ylim(0.5, 1.0)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "leakage_audit.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
