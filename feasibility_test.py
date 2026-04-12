"""
Feasibility test: Can a linear probe on frozen ESM-2 embeddings predict binding affinity?

Dataset: Davis (kinase-drug binding affinities)
Protein repr: Frozen ESM-2 mean-pooled embeddings
Ligand repr: Morgan fingerprints (2048-bit)
Baseline: Amino acid composition + physicochemical descriptors
Probe: Ridge regression (linear layer)
"""

import os
import json
import urllib.request
import pickle
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRATCH = "/ssd_scratch/akr"
DATA_DIR = os.path.join(SCRATCH, "data", "davis")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Set torch hub dir so ESM model downloads go to scratch
os.environ["TORCH_HOME"] = CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Step 1: Download Davis dataset ─────────────────────────────────────────────
def download_davis():
    base_url = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/"
    files = {
        "ligands_can.txt": "ligands_can.txt",
        "proteins.txt": "proteins.txt",
        "Y": "Y",
    }
    for fname, local in files.items():
        path = os.path.join(DATA_DIR, local)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, path)
        else:
            print(f"Already have {fname}")


def load_davis():
    with open(os.path.join(DATA_DIR, "proteins.txt")) as f:
        proteins = json.load(f)
    with open(os.path.join(DATA_DIR, "ligands_can.txt")) as f:
        ligands = json.load(f)
    with open(os.path.join(DATA_DIR, "Y"), "rb") as f:
        Y = pickle.load(f, encoding="latin1")
    affinity_matrix = np.array(Y).T  # shape: (n_proteins, n_ligands)

    protein_names = list(proteins.keys())
    ligand_names = list(ligands.keys())

    rows = []
    for i, pname in enumerate(protein_names):
        for j, lname in enumerate(ligand_names):
            kd = affinity_matrix[i, j]
            if kd > 0:
                pKd = -np.log10(kd / 1e9)
                rows.append({
                    "protein_name": pname,
                    "sequence": proteins[pname],
                    "ligand_name": lname,
                    "smiles": ligands[lname],
                    "Kd": kd,
                    "pKd": pKd,
                })

    df = pd.DataFrame(rows)
    print(f"Davis dataset: {len(df)} pairs, {df['protein_name'].nunique()} proteins, "
          f"{df['ligand_name'].nunique()} ligands")
    print(f"pKd range: [{df['pKd'].min():.2f}, {df['pKd'].max():.2f}], mean={df['pKd'].mean():.2f}")
    return df


# ── Step 2: Ligand fingerprints ────────────────────────────────────────────────
def compute_ligand_fingerprints(df, n_bits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    unique_smiles = df[["ligand_name", "smiles"]].drop_duplicates()
    fp_dict = {}
    for _, row in unique_smiles.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"WARNING: Could not parse SMILES for {row['ligand_name']}")
            fp_dict[row["ligand_name"]] = np.zeros(n_bits)
            continue
        fp_dict[row["ligand_name"]] = gen.GetFingerprintAsNumPy(mol).astype(np.float64)

    print(f"Computed Morgan FPs for {len(fp_dict)} ligands ({n_bits}-bit, radius={radius})")
    return fp_dict


# ── Step 3: ESM-2 embeddings ──────────────────────────────────────────────────
def extract_esm2_embeddings(df, model_name="esm2_t6_8M_UR50D"):
    embed_path = os.path.join(EMBED_DIR, f"{model_name}_davis.pt")
    if os.path.exists(embed_path):
        print(f"Loading cached embeddings from {embed_path}")
        return torch.load(embed_path, map_location="cpu", weights_only=True)

    import esm

    print(f"Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    unique_prots = df[["protein_name", "sequence"]].drop_duplicates()
    embed_dict = {}

    print(f"Extracting embeddings for {len(unique_prots)} proteins...")
    for idx, (_, row) in enumerate(unique_prots.iterrows()):
        name = row["protein_name"]
        seq = row["sequence"]

        # ESM-2 max tokens = 1022 residues (+ BOS + EOS = 1024)
        if len(seq) > 1022:
            seq = seq[:1022]

        data = [(name, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(num_layers + 1)))

        # Store mean-pooled embedding for each layer (exclude BOS/EOS)
        seq_len = len(seq)
        layer_embeds = {}
        for layer_idx in range(num_layers + 1):
            rep = results["representations"][layer_idx]
            # tokens: [BOS, res1, res2, ..., resN, EOS] -> slice [1:seq_len+1]
            residue_rep = rep[0, 1:seq_len + 1, :]
            mean_rep = residue_rep.mean(dim=0).cpu()
            layer_embeds[layer_idx] = mean_rep

        embed_dict[name] = layer_embeds

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique_prots)}] {name} (len={len(seq)})")

    torch.save(embed_dict, embed_path)
    print(f"Saved embeddings to {embed_path}")
    return embed_dict


# ── Step 4: Baseline protein features ─────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDRO = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# Residue molecular weights (Da)
MW = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75,
    'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149, 'N': 132,
    'P': 115, 'Q': 146, 'R': 174, 'S': 105, 'T': 119, 'V': 117,
    'W': 204, 'Y': 181,
}

AROMATIC = set("FWY")
POSITIVE = set("RKH")
NEGATIVE = set("DE")


def compute_baseline_features(sequence):
    seq = [aa for aa in sequence if aa in AMINO_ACIDS]
    n = len(seq)
    if n == 0:
        return np.zeros(28)

    # AA composition (20 features)
    aa_comp = np.array([seq.count(aa) / n for aa in AMINO_ACIDS])

    # Physicochemical descriptors (8 features)
    avg_hydro = np.mean([HYDRO.get(aa, 0) for aa in seq])
    avg_mw = np.mean([MW.get(aa, 0) for aa in seq])
    frac_aromatic = sum(1 for aa in seq if aa in AROMATIC) / n
    frac_positive = sum(1 for aa in seq if aa in POSITIVE) / n
    frac_negative = sum(1 for aa in seq if aa in NEGATIVE) / n
    frac_charged = frac_positive + frac_negative
    net_charge = frac_positive - frac_negative
    log_len = np.log(n)

    physchem = np.array([avg_hydro, avg_mw, frac_aromatic, frac_positive,
                         frac_negative, frac_charged, net_charge, log_len])

    return np.concatenate([aa_comp, physchem])


# ── Step 5: Build feature matrices and run probes ─────────────────────────────
def build_features(df, embed_dict, fp_dict, layer=-1):
    """Build feature matrices for ESM-2 probe and baseline."""
    if layer == -1:
        # Use last layer
        sample_prot = next(iter(embed_dict.values()))
        layer = max(sample_prot.keys())

    esm_features = []
    baseline_features = []
    ligand_features = []
    targets = []

    for _, row in df.iterrows():
        pname = row["protein_name"]
        lname = row["ligand_name"]

        esm_emb = embed_dict[pname][layer].numpy()
        fp = fp_dict[lname]
        baseline = compute_baseline_features(row["sequence"])

        esm_features.append(np.concatenate([esm_emb, fp]))
        baseline_features.append(np.concatenate([baseline, fp]))
        ligand_features.append(fp)
        targets.append(row["pKd"])

    return (np.array(esm_features), np.array(baseline_features),
            np.array(ligand_features), np.array(targets))


def concordance_index(y_true, y_pred):
    """Compute concordance index (CI) in O(n log n) using sorted groups + searchsorted."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(y_true)
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]

    n = len(y_true_s)
    total_pairs = n * (n - 1) / 2
    unique, counts = np.unique(y_true_s, return_counts=True)
    tied_true = sum(c * (c - 1) / 2 for c in counts)
    comparable = total_pairs - tied_true
    if comparable == 0:
        return 0.0

    # Group by unique y_true values. For each group, count concordant/tied
    # pairs against all previous groups using a sorted array + searchsorted.
    concordant = 0.0
    tied_pred = 0.0
    prev_sorted = np.array([], dtype=np.float64)

    split_indices = np.searchsorted(y_true_s, unique, side='right')
    start = 0
    for end in split_indices:
        group_preds = y_pred_s[start:end]
        if len(prev_sorted) > 0:
            for gp in group_preds:
                # Number of previous preds strictly less than gp
                idx = np.searchsorted(prev_sorted, gp, side='left')
                concordant += idx
                # Number of previous preds equal to gp
                idx_right = np.searchsorted(prev_sorted, gp, side='right')
                tied_pred += (idx_right - idx)
        # Merge group into sorted previous predictions
        prev_sorted = np.sort(np.concatenate([prev_sorted, group_preds]))
        start = end

    return (concordant + 0.5 * tied_pred) / comparable


def evaluate_probe(X_train, y_train, X_test, y_test, name="Probe"):
    """Train Ridge regression and evaluate."""
    model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r, p_val = stats.pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ci = concordance_index(y_test, y_pred)

    print(f"  {name:30s} | r={r:.4f} | MSE={mse:.4f} | R²={r2:.4f} | CI={ci:.4f}")
    return {"name": name, "pearson_r": r, "p_value": p_val, "mse": mse, "r2": r2,
            "ci": ci, "alpha": model.alpha_}


def run_layer_analysis(df, embed_dict, fp_dict, train_idx, test_idx):
    """Run probe for each ESM-2 layer."""
    sample_prot = next(iter(embed_dict.values()))
    num_layers = max(sample_prot.keys())

    print(f"\n{'='*70}")
    print(f"Layer-wise analysis (layers 0-{num_layers})")
    print(f"{'='*70}")

    results = []
    for layer in range(num_layers + 1):
        X_esm, _, _, y = build_features(df, embed_dict, fp_dict, layer=layer)
        X_train, X_test = X_esm[train_idx], X_esm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        r = evaluate_probe(X_train, y_train, X_test, y_test, name=f"Layer {layer}")
        r["layer"] = layer
        results.append(r)

    best = max(results, key=lambda x: x["pearson_r"])
    print(f"\nBest layer: {best['layer']} (r={best['pearson_r']:.4f})")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Get data
    print("="*70)
    print("Step 1: Downloading and loading Davis dataset")
    print("="*70)
    download_davis()
    df = load_davis()

    # 2. Compute ligand fingerprints
    print(f"\n{'='*70}")
    print("Step 2: Computing ligand fingerprints")
    print("="*70)
    fp_dict = compute_ligand_fingerprints(df)

    # 3. Extract ESM-2 embeddings
    print(f"\n{'='*70}")
    print("Step 3: Extracting ESM-2 embeddings")
    print("="*70)
    embed_dict = extract_esm2_embeddings(df, model_name="esm2_t6_8M_UR50D")

    # 4. Train/test split
    print(f"\n{'='*70}")
    print("Step 4: Train/test split and linear probing")
    print("="*70)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Build features (last layer)
    X_esm, X_baseline, X_ligand, y = build_features(df, embed_dict, fp_dict)
    X_esm_train, X_esm_test = X_esm[train_idx], X_esm[test_idx]
    X_bl_train, X_bl_test = X_baseline[train_idx], X_baseline[test_idx]
    X_lig_train, X_lig_test = X_ligand[train_idx], X_ligand[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 5. Run all probes
    print(f"\n{'='*70}")
    print("Results: Linear probes on test set")
    print("="*70)

    results = []

    # Main comparison
    results.append(evaluate_probe(X_esm_train, y_train, X_esm_test, y_test,
                                  "ESM-2 (8M) + Morgan FP"))
    results.append(evaluate_probe(X_bl_train, y_train, X_bl_test, y_test,
                                  "AA Composition + Morgan FP"))

    # Controls
    results.append(evaluate_probe(X_lig_train, y_train, X_lig_test, y_test,
                                  "Morgan FP only (ligand-only)"))

    # Protein-only controls
    X_esm_prot = X_esm[:, :X_esm.shape[1] - 2048]  # strip ligand FP
    results.append(evaluate_probe(X_esm_prot[train_idx], y_train,
                                  X_esm_prot[test_idx], y_test,
                                  "ESM-2 only (protein-only)"))

    X_bl_prot = X_baseline[:, :X_baseline.shape[1] - 2048]
    results.append(evaluate_probe(X_bl_prot[train_idx], y_train,
                                  X_bl_prot[test_idx], y_test,
                                  "AA Comp only (protein-only)"))

    # Shuffle control
    y_shuffled = y_train.copy()
    np.random.seed(42)
    np.random.shuffle(y_shuffled)
    results.append(evaluate_probe(X_esm_train, y_shuffled, X_esm_test, y_test,
                                  "ESM-2 + FP (SHUFFLED labels)"))

    # 6. Layer-wise analysis
    layer_results = run_layer_analysis(df, embed_dict, fp_dict, train_idx, test_idx)

    # 7. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    esm_r = results[0]["pearson_r"]
    bl_r = results[1]["pearson_r"]
    delta = esm_r - bl_r
    print(f"ESM-2 + FP Pearson r:      {esm_r:.4f}")
    print(f"Baseline + FP Pearson r:   {bl_r:.4f}")
    print(f"Delta (PLM contribution):  {delta:+.4f}")
    print()
    if delta > 0.05:
        print(">> ESM-2 embeddings provide meaningful additional signal over AA composition.")
    elif delta > 0:
        print(">> ESM-2 embeddings provide marginal additional signal over AA composition.")
    else:
        print(">> ESM-2 embeddings do NOT outperform the AA composition baseline.")

    if esm_r > 0.5:
        print(">> Binding affinity signal IS linearly extractable from frozen ESM-2 embeddings.")
    elif esm_r > 0.3:
        print(">> Weak but non-trivial affinity signal detected. Consider nonlinear probes.")
    else:
        print(">> Minimal affinity signal detected with linear probing.")


if __name__ == "__main__":
    main()
