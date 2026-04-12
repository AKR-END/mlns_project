"""
Interpretability analysis: What do protein language models learn about binding?

1. Binding pocket overlap — do attention weights localize to crystallographic
   binding pocket residues? (Uses PDBbind *_pocket.pdb files)
2. Gradient attribution — per-residue gradient importance vs attention weights
3. Amino acid type preference — which residue types are attended to?
4. Enrichment statistics — AUC-ROC of attention as a pocket classifier
5. Attention is ligand-independent — the model identifies "generally important"
   residues, not ligand-specific contacts (architectural observation)

Requires: PDBbind v2016 with pocket PDB files, trained attention model.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/ssd_scratch/akr"
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
PDBBIND_DIR = os.path.join(SCRATCH, "data", "pdbbind_v2016", "v2016")
os.environ["TORCH_HOME"] = CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import concordance_index
from ablation_study import (
    ESM_CONFIGS, load_esm_model, AttentionPooledMLP, residue_collate_fn,
)
from Bio.PDB.Polypeptide import protein_letters_3to1


# ═══════════════════════════════════════════════════════════════════════════════
# PDB parsing utilities
# ═══════════════════════════════════════════════════════════════════════════════

def parse_protein_residues(pdb_path):
    """Parse protein PDB, return list of (chain, resseq, aa_letter) and sequence.
    Only first model, standard amino acids, deduplicated by (chain, resseq).
    """
    residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL") and len(residues) > 0:
                break  # first model only
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            key = (chain, resseq, icode)
            if key not in seen and resname in protein_letters_3to1:
                seen.add(key)
                residues.append((chain, resseq, icode,
                                 protein_letters_3to1[resname]))
    seq = "".join(r[3] for r in residues)
    return residues, seq


def parse_pocket_residue_set(pocket_pdb_path):
    """Parse pocket PDB, return set of (chain, resseq, icode) tuples."""
    pocket = set()
    with open(pocket_pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26].strip()
            pocket.add((chain, resseq, icode))
    return pocket


def get_pocket_mask(pdb_code):
    """Get binary mask over protein sequence: True if residue is in pocket.
    Returns: sequence (str), pocket_mask (np.array of bool), or None if files missing.
    """
    protein_pdb = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_protein.pdb")
    pocket_pdb = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_pocket.pdb")

    if not os.path.exists(protein_pdb) or not os.path.exists(pocket_pdb):
        return None, None

    residues, seq = parse_protein_residues(protein_pdb)
    pocket_set = parse_pocket_residue_set(pocket_pdb)

    mask = np.array([(r[0], r[1], r[2]) in pocket_set for r in residues])
    return seq, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Load PDBbind test set
# ═══════════════════════════════════════════════════════════════════════════════

def load_pdbbind_codes():
    """Load PDB codes and affinities from core set index."""
    index_file = os.path.join(PDBBIND_DIR, "index", "INDEX_core_data.2016")
    codes, affinities = [], []
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            codes.append(parts[0])
            affinities.append(float(parts[3]))
    return codes, np.array(affinities)


# ═══════════════════════════════════════════════════════════════════════════════
# Extract attention weights and gradient attributions
# ═══════════════════════════════════════════════════════════════════════════════

def get_attention_weights(model, residue_embs):
    """Extract attention weights from AttentionPooledMLP.
    Args: model, residue_embs: tensor (seq_len, embed_dim)
    Returns: attention weights (seq_len,) as numpy array
    """
    model.eval()
    embs = residue_embs.unsqueeze(0).to(DEVICE)
    seq_len = embs.shape[1]
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        scores = model.attn(embs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
    return weights[0].cpu().numpy()


def get_gradient_attribution(model, residue_embs, ligand_fp):
    """Compute per-residue gradient importance: ||d(pred)/d(residue_i)||.
    Returns: gradient importance (seq_len,) as numpy array
    """
    model.eval()
    embs = residue_embs.unsqueeze(0).to(DEVICE).requires_grad_(True)
    lig = torch.FloatTensor(ligand_fp).unsqueeze(0).to(DEVICE)
    seq_len = embs.shape[1]
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=DEVICE)

    pred = model(embs, lig, mask)
    pred.backward()

    # L2 norm of gradient per residue
    grad_importance = embs.grad[0].norm(dim=-1).cpu().numpy()
    # Normalize to sum to 1 (like attention)
    grad_importance = grad_importance / (grad_importance.sum() + 1e-10)
    return grad_importance


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def pocket_overlap_analysis(model, esm_residue_dict, morgan_dict, pdb_codes,
                            sequences, smiles_list):
    """Compute attention vs pocket overlap for all PDBbind complexes."""

    print("\n" + "=" * 70)
    print("1. BINDING POCKET OVERLAP ANALYSIS")
    print("=" * 70)

    results = []
    all_attn_pocket = []     # attention weights for pocket residues
    all_attn_nonpocket = []  # attention weights for non-pocket residues

    for i, code in enumerate(pdb_codes):
        seq_pocket, pocket_mask = get_pocket_mask(code)
        if seq_pocket is None or pocket_mask is None:
            continue

        if code not in esm_residue_dict:
            continue

        residue_embs = esm_residue_dict[code]
        seq_len = residue_embs.shape[0]

        # Truncate pocket_mask to match ESM embedding length
        if len(pocket_mask) > seq_len:
            pocket_mask = pocket_mask[:seq_len]
        elif len(pocket_mask) < seq_len:
            # Padding needed — skip this protein (sequence mismatch)
            continue

        n_pocket = pocket_mask.sum()
        if n_pocket == 0 or n_pocket == len(pocket_mask):
            continue  # skip trivial cases

        # Get attention weights
        attn_w = get_attention_weights(model, residue_embs)
        if len(attn_w) != len(pocket_mask):
            continue

        # Get gradient attribution
        lig_fp = morgan_dict.get(code)
        if lig_fp is not None:
            grad_w = get_gradient_attribution(model, residue_embs, lig_fp)
        else:
            grad_w = None

        # Collect pocket vs non-pocket attention
        all_attn_pocket.extend(attn_w[pocket_mask].tolist())
        all_attn_nonpocket.extend(attn_w[~pocket_mask].tolist())

        # Top-K precision (K = number of pocket residues)
        k = n_pocket
        top_k_idx = np.argsort(attn_w)[-k:]
        top_k_in_pocket = pocket_mask[top_k_idx].sum()
        precision_at_k = top_k_in_pocket / k
        expected_precision = n_pocket / len(pocket_mask)
        enrichment = precision_at_k / expected_precision if expected_precision > 0 else 0

        # AUC-ROC: can attention predict pocket membership?
        try:
            auroc = roc_auc_score(pocket_mask.astype(int), attn_w)
        except ValueError:
            auroc = 0.5

        # Gradient AUC-ROC
        grad_auroc = 0.5
        if grad_w is not None and len(grad_w) == len(pocket_mask):
            try:
                grad_auroc = roc_auc_score(pocket_mask.astype(int), grad_w)
            except ValueError:
                grad_auroc = 0.5

        # Attention-gradient correlation
        attn_grad_corr = 0.0
        if grad_w is not None and len(grad_w) == len(attn_w):
            attn_grad_corr, _ = stats.spearmanr(attn_w, grad_w)

        results.append({
            "code": code,
            "seq_len": seq_len,
            "n_pocket": int(n_pocket),
            "pocket_frac": n_pocket / len(pocket_mask),
            "precision_at_k": precision_at_k,
            "enrichment": enrichment,
            "attn_auroc": auroc,
            "grad_auroc": grad_auroc,
            "attn_grad_corr": attn_grad_corr,
        })

    print(f"\nAnalyzed {len(results)} complexes")

    # Aggregate statistics
    enrichments = [r["enrichment"] for r in results]
    aurocs = [r["attn_auroc"] for r in results]
    grad_aurocs = [r["grad_auroc"] for r in results]
    ag_corrs = [r["attn_grad_corr"] for r in results]

    print(f"\n--- Pocket Enrichment ---")
    print(f"  Top-K precision enrichment: {np.mean(enrichments):.2f}x "
          f"+/- {np.std(enrichments):.2f} (1.0 = random)")
    print(f"  Enrichment > 1.0 in {sum(e > 1 for e in enrichments)}/{len(enrichments)} "
          f"complexes ({100*sum(e > 1 for e in enrichments)/len(enrichments):.0f}%)")

    print(f"\n--- AUC-ROC (attention as pocket classifier) ---")
    print(f"  Attention AUC-ROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f} "
          f"(0.5 = random)")
    print(f"  Gradient AUC-ROC:  {np.mean(grad_aurocs):.4f} +/- {np.std(grad_aurocs):.4f}")

    print(f"\n--- Attention vs Gradient correlation ---")
    print(f"  Spearman r: {np.mean(ag_corrs):.4f} +/- {np.std(ag_corrs):.4f}")

    # Wilcoxon test: pocket vs non-pocket attention
    stat, p = stats.mannwhitneyu(all_attn_pocket, all_attn_nonpocket,
                                  alternative="greater")
    mean_pocket = np.mean(all_attn_pocket)
    mean_nonpocket = np.mean(all_attn_nonpocket)
    print(f"\n--- Pocket vs Non-pocket Attention ---")
    print(f"  Mean attention (pocket):     {mean_pocket:.6f}")
    print(f"  Mean attention (non-pocket): {mean_nonpocket:.6f}")
    print(f"  Ratio: {mean_pocket/mean_nonpocket:.2f}x")
    print(f"  Mann-Whitney U p-value: {p:.2e}")

    return results, all_attn_pocket, all_attn_nonpocket


def amino_acid_preference(model, esm_residue_dict, pdb_codes, sequences):
    """Analyze which amino acid types receive higher attention."""

    print("\n" + "=" * 70)
    print("2. AMINO ACID TYPE PREFERENCE")
    print("=" * 70)

    aa_attention = {}  # {aa: [list of attention weights]}

    for i, code in enumerate(pdb_codes):
        if code not in esm_residue_dict:
            continue

        residue_embs = esm_residue_dict[code]
        seq = sequences[i][:residue_embs.shape[0]]

        attn_w = get_attention_weights(model, residue_embs)

        for j, aa in enumerate(seq):
            if j >= len(attn_w):
                break
            aa_attention.setdefault(aa, []).append(attn_w[j])

    # Compute mean attention per AA type, normalized by expected (uniform)
    print(f"\n{'AA':<4s} | {'N':>8s} | {'Mean Attn':>10s} | {'Enrichment':>11s}")
    print("-" * 45)

    aa_stats = {}
    for aa in sorted(aa_attention.keys()):
        weights = aa_attention[aa]
        mean_w = np.mean(weights)
        # Expected attention if uniform over all residues
        # This varies per protein, so enrichment is approximate
        aa_stats[aa] = {"mean": mean_w, "count": len(weights)}

    # Normalize enrichments: divide by global mean attention weight
    global_mean = np.mean([w for ws in aa_attention.values() for w in ws])
    for aa in sorted(aa_stats.keys(), key=lambda a: -aa_stats[a]["mean"]):
        s = aa_stats[aa]
        enrichment = s["mean"] / global_mean
        print(f"  {aa:<2s}  | {s['count']:>8d} | {s['mean']:>10.6f} | "
              f"{enrichment:>10.2f}x")
        aa_stats[aa]["enrichment"] = enrichment

    # Group by property
    hydrophobic = set("AILMFWVP")
    aromatic = set("FWY")
    charged_pos = set("RKH")
    charged_neg = set("DE")
    polar = set("STNQ")

    groups = {
        "Hydrophobic (AILMFWVP)": hydrophobic,
        "Aromatic (FWY)": aromatic,
        "Positive (RKH)": charged_pos,
        "Negative (DE)": charged_neg,
        "Polar (STNQ)": polar,
        "Others (GC)": set("GC"),
    }

    print(f"\n--- By Residue Property ---")
    for gname, gset in groups.items():
        weights = [w for aa in gset if aa in aa_attention for w in aa_attention[aa]]
        if weights:
            enrichment = np.mean(weights) / global_mean
            print(f"  {gname:<25s}: enrichment = {enrichment:.2f}x "
                  f"(n={len(weights)})")

    return aa_stats


def gradient_vs_attention_analysis(model, esm_residue_dict, morgan_dict,
                                    pdb_codes, sequences):
    """Compare gradient attribution with attention weights."""

    print("\n" + "=" * 70)
    print("3. GRADIENT ATTRIBUTION vs ATTENTION")
    print("=" * 70)

    correlations = []
    agreement_at_10 = []  # fraction of top-10 attended that are also top-10 gradient

    for code in pdb_codes:
        if code not in esm_residue_dict or code not in morgan_dict:
            continue

        residue_embs = esm_residue_dict[code]
        lig_fp = morgan_dict[code]

        attn_w = get_attention_weights(model, residue_embs)
        grad_w = get_gradient_attribution(model, residue_embs, lig_fp)

        if len(attn_w) != len(grad_w):
            continue

        corr, _ = stats.spearmanr(attn_w, grad_w)
        correlations.append(corr)

        # Top-10 agreement
        k = min(10, len(attn_w))
        top_attn = set(np.argsort(attn_w)[-k:])
        top_grad = set(np.argsort(grad_w)[-k:])
        overlap = len(top_attn & top_grad) / k
        agreement_at_10.append(overlap)

    print(f"\n  Spearman correlation: {np.mean(correlations):.4f} "
          f"+/- {np.std(correlations):.4f}")
    print(f"  Top-10 agreement:    {np.mean(agreement_at_10):.4f} "
          f"+/- {np.std(agreement_at_10):.4f}")

    if np.mean(correlations) > 0.5:
        print("  -> Attention and gradients AGREE: attention is faithful "
              "to prediction mechanism")
    elif np.mean(correlations) > 0.2:
        print("  -> Moderate agreement: attention partially reflects "
              "prediction-relevant residues")
    else:
        print("  -> Attention and gradients DISAGREE: attention is not "
              "faithful — it highlights different residues than what "
              "drives predictions")

    return correlations, agreement_at_10


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(overlap_results, all_attn_pocket, all_attn_nonpocket,
                 aa_stats, correlations, agreement):

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Interpretability: What Does the Model Learn About Binding?",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Plot 1: Enrichment distribution ──
    ax1 = fig.add_subplot(2, 3, 1)
    enrichments = [r["enrichment"] for r in overlap_results]
    ax1.hist(enrichments, bins=20, color="#26A69A", alpha=0.8, edgecolor="white")
    ax1.axvline(x=1.0, color="red", linestyle="--", label="Random (1.0x)")
    ax1.axvline(x=np.mean(enrichments), color="blue", linestyle="-",
                label=f"Mean ({np.mean(enrichments):.2f}x)")
    ax1.set_xlabel("Pocket Enrichment")
    ax1.set_ylabel("Count")
    ax1.set_title("Attention → Pocket Enrichment", fontweight="bold")
    ax1.legend(fontsize=8)

    # ── Plot 2: AUC-ROC distribution ──
    ax2 = fig.add_subplot(2, 3, 2)
    aurocs = [r["attn_auroc"] for r in overlap_results]
    grad_aurocs = [r["grad_auroc"] for r in overlap_results]
    ax2.hist(aurocs, bins=20, alpha=0.7, color="#42A5F5", label="Attention",
             edgecolor="white")
    ax2.hist(grad_aurocs, bins=20, alpha=0.7, color="#FF7043", label="Gradient",
             edgecolor="white")
    ax2.axvline(x=0.5, color="red", linestyle="--", label="Random (0.5)")
    ax2.set_xlabel("AUC-ROC (pocket prediction)")
    ax2.set_ylabel("Count")
    ax2.set_title("Attention/Gradient as Pocket Classifier", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Plot 3: Pocket vs non-pocket attention distribution ──
    ax3 = fig.add_subplot(2, 3, 3)
    bins = np.linspace(0, max(np.percentile(all_attn_pocket, 99),
                              np.percentile(all_attn_nonpocket, 99)), 50)
    ax3.hist(all_attn_nonpocket, bins=bins, alpha=0.7, density=True,
             color="#78909C", label="Non-pocket", edgecolor="white")
    ax3.hist(all_attn_pocket, bins=bins, alpha=0.7, density=True,
             color="#EF5350", label="Pocket", edgecolor="white")
    ax3.set_xlabel("Attention Weight")
    ax3.set_ylabel("Density")
    ax3.set_title("Attention Distribution: Pocket vs Non-pocket",
                  fontweight="bold")
    ax3.legend(fontsize=8)

    # ── Plot 4: AA type enrichment ──
    ax4 = fig.add_subplot(2, 3, 4)
    aas = sorted(aa_stats.keys(),
                 key=lambda a: -aa_stats[a].get("enrichment", 1.0))
    enrichments_aa = [aa_stats[a].get("enrichment", 1.0) for a in aas]
    colors = []
    hydrophobic = set("AILMFWVP")
    charged = set("RKHDE")
    polar = set("STNQ")
    for aa in aas:
        if aa in hydrophobic:
            colors.append("#FF7043")
        elif aa in charged:
            colors.append("#42A5F5")
        elif aa in polar:
            colors.append("#66BB6A")
        else:
            colors.append("#78909C")

    ax4.barh(range(len(aas)), enrichments_aa, color=colors, alpha=0.85)
    ax4.set_yticks(range(len(aas)))
    ax4.set_yticklabels(aas, fontsize=8)
    ax4.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Attention Enrichment (vs uniform)")
    ax4.set_title("Amino Acid Attention Preference", fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF7043", label="Hydrophobic"),
        Patch(facecolor="#42A5F5", label="Charged"),
        Patch(facecolor="#66BB6A", label="Polar"),
        Patch(facecolor="#78909C", label="Other"),
    ]
    ax4.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax4.invert_yaxis()

    # ── Plot 5: Attention-Gradient correlation ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(correlations, bins=20, color="#AB47BC", alpha=0.8, edgecolor="white")
    ax5.axvline(x=np.mean(correlations), color="blue", linestyle="-",
                label=f"Mean ({np.mean(correlations):.3f})")
    ax5.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Spearman Correlation (Attention vs Gradient)")
    ax5.set_ylabel("Count")
    ax5.set_title("Attention Faithfulness", fontweight="bold")
    ax5.legend(fontsize=8)

    # ── Plot 6: Example protein with pocket overlay ──
    ax6 = fig.add_subplot(2, 3, 6)
    # Find a good example (medium length, decent enrichment)
    good = [r for r in overlap_results
            if 200 < r["seq_len"] < 500 and r["enrichment"] > 1.5]
    if good:
        ex = sorted(good, key=lambda r: -r["attn_auroc"])[0]
    else:
        ex = overlap_results[0]

    code = ex["code"]
    from pdbbind_eval import load_pdbbind_test
    # Just reparse this one protein
    seq_pocket, pocket_mask = get_pocket_mask(code)
    if seq_pocket is not None and code in esm_residue_dict_global:
        residue_embs = esm_residue_dict_global[code]
        attn_w = get_attention_weights(model_global, residue_embs)
        sl = min(len(attn_w), len(pocket_mask))
        attn_w = attn_w[:sl]
        pocket_mask_trunc = pocket_mask[:sl]

        # Plot attention
        ax6.bar(range(sl), attn_w, width=1.0, color="#26A69A", alpha=0.6,
                label="Attention")
        # Overlay pocket regions
        pocket_pos = np.where(pocket_mask_trunc)[0]
        for p in pocket_pos:
            ax6.axvspan(p - 0.5, p + 0.5, alpha=0.15, color="red")
        ax6.bar([], [], color="red", alpha=0.3, label="Pocket residues")
        ax6.set_xlabel("Residue Position")
        ax6.set_ylabel("Attention Weight")
        ax6.set_title(f"Example: PDB {code} (AUC={ex['attn_auroc']:.3f})",
                      fontweight="bold")
        ax6.set_xlim(0, sl)
        ax6.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("interpretability.png", dpi=150, bbox_inches="tight")
    print("\nSaved interpretability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

# Globals for plotting callback
esm_residue_dict_global = None
model_global = None


def main():
    global esm_residue_dict_global, model_global

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # ── Load PDBbind test set ──
    pdb_codes, affinities = load_pdbbind_codes()
    print(f"PDBbind core set: {len(pdb_codes)} complexes")

    # Get sequences and SMILES (need to parse PDB/SDF files)
    from rdkit import Chem
    sequences, smiles_list, valid_codes, valid_affinities = [], [], [], []
    for code, aff in zip(pdb_codes, affinities):
        protein_pdb = os.path.join(PDBBIND_DIR, code, f"{code}_protein.pdb")
        sdf_file = os.path.join(PDBBIND_DIR, code, f"{code}_ligand.sdf")
        pocket_pdb = os.path.join(PDBBIND_DIR, code, f"{code}_pocket.pdb")

        if not all(os.path.exists(f) for f in [protein_pdb, sdf_file, pocket_pdb]):
            continue

        _, seq = parse_protein_residues(protein_pdb)
        if len(seq) == 0:
            continue

        try:
            supplier = Chem.SDMolSupplier(sdf_file, sanitize=True)
            mol = next(iter(supplier))
            if mol is None:
                supplier = Chem.SDMolSupplier(sdf_file, sanitize=False)
                mol = next(iter(supplier))
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue

        sequences.append(seq)
        smiles_list.append(smi)
        valid_codes.append(code)
        valid_affinities.append(aff)

    print(f"Valid complexes with pocket files: {len(valid_codes)}")

    # ── Load/extract embeddings ──
    print("\nLoading ESM-2 650M per-residue embeddings...")
    from pdbbind_eval import extract_esm2_residue, compute_morgan
    esm_residue = extract_esm2_residue(sequences, valid_codes, "pdbbind_test")
    morgan = compute_morgan(smiles_list, valid_codes)

    esm_residue_dict_global = esm_residue

    # ── Train attention model (on jglaser, or load if available) ──
    # For interpretability, we train on Davis (faster) and analyze on PDBbind
    print("\nTraining attention model on Davis...")
    from feasibility_test import download_davis, load_davis, compute_ligand_fingerprints
    from ablation_study import extract_residue_embeddings
    from sklearn.model_selection import train_test_split

    download_davis()
    davis_df = load_davis()
    davis_morgan = compute_ligand_fingerprints(davis_df)
    davis_residue = extract_residue_embeddings(davis_df, "650M")

    indices = np.arange(len(davis_df))
    train_val_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2,
                                          random_state=42)

    from ablation_study import train_attention_mlp
    cfg = ESM_CONFIGS["650M"]
    model = train_attention_mlp(
        davis_df, train_idx, val_idx, davis_residue, davis_morgan,
        embed_dim=cfg["dim"], ligand_dim=2048)

    model_global = model

    # ── Run analyses ──

    # 1. Pocket overlap
    overlap_results, attn_pocket, attn_nonpocket = pocket_overlap_analysis(
        model, esm_residue, morgan, valid_codes, sequences, smiles_list)

    # 2. Amino acid preference
    aa_stats = amino_acid_preference(model, esm_residue, valid_codes, sequences)

    # 3. Gradient vs attention
    correlations, agreement = gradient_vs_attention_analysis(
        model, esm_residue, morgan, valid_codes, sequences)

    # 4. Key observation: attention is ligand-independent
    print("\n" + "=" * 70)
    print("4. ATTENTION IS LIGAND-INDEPENDENT")
    print("=" * 70)
    print("  Our attention layer: attn = Linear(embed_dim, 1)")
    print("  -> Attention weights depend ONLY on protein residue embeddings")
    print("  -> Same protein gets same attention regardless of which ligand binds")
    print("  -> The model identifies 'generally important binding residues'")
    print("     not ligand-specific contacts")
    print("  -> This is expected: frozen PLM embeddings don't encode ligand info")

    # ── Plot everything ──
    plot_results(overlap_results, attn_pocket, attn_nonpocket,
                 aa_stats, correlations, agreement)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mean_enrichment = np.mean([r["enrichment"] for r in overlap_results])
    mean_auroc = np.mean([r["attn_auroc"] for r in overlap_results])
    mean_corr = np.mean(correlations)

    print(f"\n  Pocket enrichment:    {mean_enrichment:.2f}x (>1 = better than random)")
    print(f"  Pocket AUC-ROC:       {mean_auroc:.4f} (>0.5 = better than random)")
    print(f"  Attn-Gradient corr:   {mean_corr:.4f}")

    if mean_enrichment > 1.5 and mean_auroc > 0.6:
        print("\n  CONCLUSION: Attention localizes to binding pockets.")
        print("  ESM-2 embeddings encode binding-site-relevant features")
        print("  that are recoverable by a simple learned attention layer,")
        print("  despite never seeing 3D structure during pretraining.")
    elif mean_enrichment > 1.0 and mean_auroc > 0.55:
        print("\n  CONCLUSION: Weak but significant pocket localization.")
        print("  The model partially captures binding site information")
        print("  but also relies on global protein features.")
    else:
        print("\n  CONCLUSION: Attention does NOT localize to binding pockets.")
        print("  The model uses global protein features for prediction,")
        print("  not local binding site recognition.")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
