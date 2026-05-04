"""
Shared utilities for all experiments.

Data splits:
  Train: jglaser/binding_affinity (HuggingFace)
  Val:   PDBbind v2016 refined set minus core (3,767 complexes)
  Test:  PDBbind v2016 core set / CASF-2016 (290 complexes)

Provides: data loading, ESM-2 embedding extraction, Morgan FP computation,
model definitions, training loops, evaluation metrics.
"""

import os
import sys
import numpy as np
import torch

# Silence RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from collections import OrderedDict

# ── Paths ─────────────────────────────────────────────────────────────────────
# Scratch (large/ephemeral): embeddings, checkpoints, model caches.
SCRATCH = os.environ.get("MLNS_SCRATCH", "/ssd_scratch/akr")
DATA_DIR = os.path.join(SCRATCH, "data")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
CKPT_DIR = os.path.join(SCRATCH, "checkpoints")
PDBBIND_DIR = os.path.join(DATA_DIR, "pdbbind_v2016", "v2016")
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Project root: the directory containing this file's parent (if utils.py is
# in src/) or the directory containing this file (if utils.py is at root).
# This makes all the scripts portable regardless of where the repo is cloned.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "src" else _THIS_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
DATA_EXTRA_DIR = os.path.join(PROJECT_ROOT, "data_extra")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = os.path.join(SCRATCH, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESM_CONFIGS = OrderedDict([
    ("8M",   {"hub": "esm2_t6_8M_UR50D",   "layers": 6,  "dim": 320}),
    ("35M",  {"hub": "esm2_t12_35M_UR50D",  "layers": 12, "dim": 480}),
    ("650M", {"hub": "esm2_t33_650M_UR50D", "layers": 33, "dim": 1280}),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _content_name(prefix, s):
    """Deterministic content-addressed name: prefix_<md5(s)[:12]>.
    Invariant to dict/set iteration order or process-level non-determinism.
    """
    import hashlib
    return f"{prefix}_{hashlib.md5(s.encode()).hexdigest()[:12]}"


def load_jglaser(num_samples=None):
    """Load jglaser/binding_affinity from HuggingFace.
    Returns: sequences (list), smiles (list), targets (np.array), prot_names, lig_names

    Names are content-addressed hashes of the sequence/SMILES string so they
    remain stable across separate process invocations.

    Newer `datasets` versions removed support for script-based datasets, which
    breaks the standard load_dataset path for jglaser/binding_affinity. We
    first try to read the cached parquet shipped with the snapshot; only fall
    back to load_dataset if the parquet is missing.
    """
    import glob
    hf_cache = os.path.join(SCRATCH, "hf_cache")
    parquet_glob = os.path.join(
        hf_cache, "datasets--jglaser--binding_affinity",
        "snapshots", "*", "data", "*.parquet")
    parquet_files = sorted(glob.glob(parquet_glob))

    if parquet_files:
        import pandas as pd
        print(f"Loading jglaser/binding_affinity from cached parquet: "
              f"{parquet_files[0]}")
        cols = ["seq", "smiles_can", "neg_log10_affinity_M"]
        if num_samples is None:
            df = pd.read_parquet(parquet_files[0], columns=cols)
        else:
            # Streaming row-group read so we don't materialize 1.4M rows.
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_files[0])
            chunks, taken = [], 0
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=cols).to_pandas()
                if taken + len(t) >= num_samples:
                    chunks.append(t.iloc[: num_samples - taken])
                    taken = num_samples
                    break
                chunks.append(t)
                taken += len(t)
            df = pd.concat(chunks, ignore_index=True)
        print(f"  Read {len(df)} rows")
        sequences = df["seq"].tolist()
        smiles = df["smiles_can"].tolist()
        targets = df["neg_log10_affinity_M"].to_numpy(dtype=np.float32)
    else:
        from datasets import load_dataset
        if num_samples is None:
            print("Loading ALL samples from jglaser/binding_affinity (HF)...")
            ds = load_dataset("jglaser/binding_affinity", split="train",
                              cache_dir=hf_cache)
        else:
            print(f"Loading {num_samples} samples from jglaser/binding_affinity (HF)...")
            ds = load_dataset("jglaser/binding_affinity",
                              split=f"train[:{num_samples}]", cache_dir=hf_cache)
        sequences = ds["seq"]
        smiles = ds["smiles_can"]
        targets = np.array(ds["neg_log10_affinity_M"], dtype=np.float32)

    valid = ~np.isnan(targets)
    sequences = [s for s, v in zip(sequences, valid) if v]
    smiles = [s for s, v in zip(smiles, valid) if v]
    targets = targets[valid]

    prot_names = [_content_name("prot", s) for s in sequences]
    lig_names = [_content_name("lig", s) for s in smiles]

    print(f"Loaded {len(targets)} samples, {len(set(prot_names))} unique proteins, "
          f"{len(set(lig_names))} unique ligands")
    return sequences, smiles, targets, prot_names, lig_names


def load_pdbbind_set(set_type="core"):
    """Load PDBbind complexes from index file.
    set_type: 'core' (290 test) or 'refined' (4057)
    Returns: pdb_codes, sequences, smiles_list, affinities
    """
    import pickle
    from rdkit import Chem
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    cache_path = os.path.join(DATA_DIR, f"pdbbind_{set_type}_parsed.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            out_codes, out_seqs, out_smiles, out_aff = pickle.load(f)
        print(f"PDBbind {set_type}: loaded {len(out_codes)} complexes (cached)")
        return out_codes, out_seqs, out_smiles, np.array(out_aff)

    if set_type == "core":
        index_file = os.path.join(PDBBIND_DIR, "index", "INDEX_core_data.2016")
    else:
        index_file = os.path.join(PDBBIND_DIR, "index", "INDEX_refined_data.2016")

    pdb_codes, affinities = [], []
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdb_codes.append(parts[0])
            affinities.append(float(parts[3]))

    parser = PDBParser(QUIET=True)
    out_codes, out_seqs, out_smiles, out_aff = [], [], [], []

    for i, (code, aff) in enumerate(zip(pdb_codes, affinities)):
        pdb_file = os.path.join(PDBBIND_DIR, code, f"{code}_protein.pdb")
        sdf_file = os.path.join(PDBBIND_DIR, code, f"{code}_ligand.sdf")
        if not os.path.exists(pdb_file) or not os.path.exists(sdf_file):
            continue
        try:
            structure = parser.get_structure(code, pdb_file)
            seq = ""
            for model in structure:
                for chain in model:
                    for residue in chain:
                        rn = residue.get_resname().strip()
                        if rn in protein_letters_3to1:
                            seq += protein_letters_3to1[rn]
                break
        except Exception:
            continue
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

        out_codes.append(code)
        out_seqs.append(seq)
        out_smiles.append(smi)
        out_aff.append(aff)

    print(f"PDBbind {set_type}: loaded {len(out_codes)}/{len(pdb_codes)} complexes")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((out_codes, out_seqs, out_smiles, out_aff), f)
    print(f"  Cached to {cache_path}")
    return out_codes, out_seqs, out_smiles, np.array(out_aff)


def load_pdbbind_val_test():
    """Load val (refined minus core) and test (core) sets.
    Returns: val_codes, val_seqs, val_smiles, val_targets,
             test_codes, test_seqs, test_smiles, test_targets
    """
    test_codes, test_seqs, test_smiles, test_targets = load_pdbbind_set("core")
    ref_codes, ref_seqs, ref_smiles, ref_targets = load_pdbbind_set("refined")

    test_set = set(test_codes)
    val_mask = [c not in test_set for c in ref_codes]
    val_codes = [c for c, m in zip(ref_codes, val_mask) if m]
    val_seqs = [s for s, m in zip(ref_seqs, val_mask) if m]
    val_smiles = [s for s, m in zip(ref_smiles, val_mask) if m]
    val_targets = ref_targets[val_mask]

    print(f"Val: {len(val_codes)} (refined-core), Test: {len(test_codes)} (core)")
    return (val_codes, val_seqs, val_smiles, val_targets,
            test_codes, test_seqs, test_smiles, test_targets)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_esm_model(hub_name):
    import esm
    ckpt = os.path.join(CACHE_DIR, "hub", "checkpoints", f"{hub_name}.pt")
    if os.path.exists(ckpt):
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            hub_name, data, None)
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet(hub_name)
    return model.eval().to(DEVICE), alphabet


def extract_esm2_embeddings(sequences, names, cache_name, label="650M",
                            per_residue=False, all_layers=False):
    """Unified ESM-2 embedding extraction.
    per_residue=False, all_layers=False: mean-pooled last layer {name: tensor(d)}
    per_residue=True:  last layer per-residue {name: tensor(L, d)}
    all_layers=True:   mean-pooled all layers {name: {layer: tensor(d)}}
    """
    cfg = ESM_CONFIGS[label]
    suffix = "residue" if per_residue else ("alllayer" if all_layers else "mean")

    # Per-residue: use sharded directory (one file per protein) to avoid OOM
    if per_residue:
        shard_dir = os.path.join(EMBED_DIR, f"esm2_{label}_{suffix}_{cache_name}")
        if os.path.isdir(shard_dir) and len(os.listdir(shard_dir)) > 0:
            print(f"Using sharded cache {label} {suffix} ({cache_name}): {shard_dir}")
            return shard_dir  # return path, not dict
        os.makedirs(shard_dir, exist_ok=True)

        model, alphabet = load_esm_model(cfg["hub"])
        batch_converter = alphabet.get_batch_converter()
        num_layers = cfg["layers"]
        unique = list(set(zip(names, sequences)))
        print(f"Extracting ESM-2 {label} {suffix} for {len(unique)} proteins (sharded)...")

        for idx, (name, seq) in enumerate(unique):
            fpath = os.path.join(shard_dir, f"{name}.pt")
            if os.path.exists(fpath):
                if (idx + 1) % 1000 == 0:
                    print(f"  [{idx+1}/{len(unique)}] (cached)")
                continue
            seq = seq[:1022].upper()
            _, _, tokens = batch_converter([(name, seq)])
            tokens = tokens.to(DEVICE)
            with torch.no_grad():
                results = model(tokens, repr_layers=[num_layers])
            sl = len(seq)
            emb = results["representations"][num_layers][0, 1:sl+1, :].cpu()
            torch.save(emb, fpath)
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(unique)}]")

        del model
        torch.cuda.empty_cache()
        print(f"Saved sharded to {shard_dir}")
        return shard_dir

    # Non-residue: use single .pt file as before
    cache_path = os.path.join(EMBED_DIR, f"esm2_{label}_{suffix}_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached {label} {suffix} ({cache_name})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    model, alphabet = load_esm_model(cfg["hub"])
    batch_converter = alphabet.get_batch_converter()
    num_layers = cfg["layers"]

    unique = list(set(zip(names, sequences)))
    embed_dict = {}
    print(f"Extracting ESM-2 {label} {suffix} for {len(unique)} proteins...")

    repr_layers = list(range(num_layers + 1)) if all_layers else [num_layers]

    for idx, (name, seq) in enumerate(unique):
        seq = seq[:1022].upper()
        _, _, tokens = batch_converter([(name, seq)])
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = model(tokens, repr_layers=repr_layers)
        sl = len(seq)

        if all_layers:
            embed_dict[name] = {l: results["representations"][l][0, 1:sl+1, :].mean(dim=0).cpu()
                                for l in range(num_layers + 1)}
        else:
            embed_dict[name] = results["representations"][num_layers][0, 1:sl+1, :].mean(dim=0).cpu()

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique)}]")

    del model
    torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    print(f"Saved to {cache_path}")
    return embed_dict


def compute_morgan(smiles_list, names, n_bits=2048, radius=2):
    """Compute Morgan FPs. Returns {name: np.array(n_bits)}."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_dict = {}
    for name, smi in zip(names, smiles_list):
        if name in fp_dict:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp_dict[name] = np.zeros(n_bits, dtype=np.float32)
        else:
            fp_dict[name] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    print(f"Computed Morgan FPs for {len(fp_dict)} ligands")
    return fp_dict


CHEMBERTA_DIM = 384
MOLFORMER_DIM = 768


def _masked_mean(hidden, mask):
    """Mean-pool hidden states over tokens, ignoring padding.
    hidden: (B, L, D), mask: (B, L). Returns (B, D)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def compute_chemberta(smiles_list, names, cache_name):
    """Compute ChemBERTa-77M-MLM embeddings via mask-aware mean pooling
    over the full token sequence (384-d). Caches to
    EMBED_DIR/chemberta_mean_{cache_name}.pt.
    """
    cache_path = os.path.join(EMBED_DIR, f"chemberta_mean_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached ChemBERTa-mean ({cache_name})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from transformers import AutoModel, AutoTokenizer
    hf_cache = os.path.join(SCRATCH, "hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", cache_dir=hf_cache)
    model = AutoModel.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", cache_dir=hf_cache)
    model = model.eval().to(DEVICE)

    unique = {}
    for nm, smi in zip(names, smiles_list):
        if nm not in unique:
            unique[nm] = smi
    u_names, u_smiles = list(unique.keys()), list(unique.values())

    embed_dict = {}
    BS = 64
    print(f"Extracting ChemBERTa (mean-pool) for {len(u_names)} ligands...")
    for i in range(0, len(u_names), BS):
        batch_smi = u_smiles[i:i+BS]
        batch_nm = u_names[i:i+BS]
        inputs = tokenizer(batch_smi, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        pooled = _masked_mean(out.last_hidden_state, inputs["attention_mask"])
        embs = pooled.cpu().numpy()
        for nm, emb in zip(batch_nm, embs):
            embed_dict[nm] = emb.astype(np.float32)
        if (i // BS) % 20 == 0:
            print(f"  [{i+len(batch_nm)}/{len(u_names)}]")

    del model; torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    print(f"Saved to {cache_path}")
    return embed_dict


def compute_molformer(smiles_list, names, cache_name):
    """Compute MoLFormer-XL embeddings via mask-aware mean pooling
    over the full token sequence (768-d). Caches to
    EMBED_DIR/molformer_{cache_name}.pt.
    """
    cache_path = os.path.join(EMBED_DIR, f"molformer_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached MolFormer ({cache_name})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from transformers import AutoModel, AutoTokenizer
    hf_cache = os.path.join(SCRATCH, "hf_cache")
    model_id = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=hf_cache)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, deterministic_eval=True,
        cache_dir=hf_cache)
    model = model.eval().to(DEVICE)

    unique = {}
    for nm, smi in zip(names, smiles_list):
        if nm not in unique:
            unique[nm] = smi
    u_names, u_smiles = list(unique.keys()), list(unique.values())

    embed_dict = {}
    BS = 64
    print(f"Extracting MolFormer (mean-pool) for {len(u_names)} ligands...")
    for i in range(0, len(u_names), BS):
        batch_smi = u_smiles[i:i+BS]
        batch_nm = u_names[i:i+BS]
        inputs = tokenizer(batch_smi, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        pooled = _masked_mean(out.last_hidden_state, inputs["attention_mask"])
        embs = pooled.cpu().numpy()
        for nm, emb in zip(batch_nm, embs):
            embed_dict[nm] = emb.astype(np.float32)
        if (i // BS) % 20 == 0:
            print(f"  [{i+len(batch_nm)}/{len(u_names)}]")

    del model; torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    print(f"Saved to {cache_path}")
    return embed_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(y_true)
    y_true_s, y_pred_s = y_true[order], y_pred[order]
    n = len(y_true_s)
    total = n * (n - 1) / 2
    unique, counts = np.unique(y_true_s, return_counts=True)
    tied = sum(c * (c - 1) / 2 for c in counts)
    comparable = total - tied
    if comparable == 0:
        return 0.0
    concordant, tied_pred = 0.0, 0.0
    prev_sorted = np.array([], dtype=np.float64)
    split_indices = np.searchsorted(y_true_s, unique, side='right')
    start = 0
    for end in split_indices:
        group = y_pred_s[start:end]
        if len(prev_sorted) > 0:
            for gp in group:
                idx = np.searchsorted(prev_sorted, gp, side='left')
                concordant += idx
                tied_pred += np.searchsorted(prev_sorted, gp, side='right') - idx
        prev_sorted = np.sort(np.concatenate([prev_sorted, group]))
        start = end
    return (concordant + 0.5 * tied_pred) / comparable


def evaluate(y_true, y_pred, name="", denorm_mean=0, denorm_std=1):
    """Evaluate predictions. Denormalizes if trained on standardized targets."""
    y_pred_dn = y_pred * denorm_std + denorm_mean
    y_true_dn = y_true * denorm_std + denorm_mean if denorm_std != 1 else y_true
    r, _ = stats.pearsonr(y_true_dn, y_pred_dn)
    mse = mean_squared_error(y_true_dn, y_pred_dn)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_dn, y_pred_dn)
    ci = concordance_index(y_true_dn, y_pred_dn)
    return {"name": name, "r": r, "rmse": rmse, "mse": mse, "r2": r2, "ci": ci}


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════

class ConcatDS(Dataset):
    """[prot_emb || lig_emb] concatenated features."""
    def __init__(self, prot_names, lig_names, targets, prot_dict, lig_dict):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.pd, self.ld = prot_dict, lig_dict
    def __len__(self): return len(self.t)
    def __getitem__(self, idx):
        p = self.pd[self.pn[idx]]
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray): l = torch.FloatTensor(l)
        if isinstance(p, np.ndarray): p = torch.FloatTensor(p)
        return torch.cat([p, l]), torch.tensor(self.t[idx], dtype=torch.float32)


class ResidueDS(Dataset):
    """Per-residue protein + ligand features. Supports sharded dir or dict."""
    def __init__(self, prot_names, lig_names, targets, prot_src, lig_dict):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.ld = lig_dict
        # prot_src can be a dict or a directory path (sharded)
        if isinstance(prot_src, str) and os.path.isdir(prot_src):
            self.shard_dir = prot_src
            self.pd = None
        else:
            self.shard_dir = None
            self.pd = prot_src
    def __len__(self): return len(self.t)
    def __getitem__(self, idx):
        pn = self.pn[idx]
        if self.shard_dir is not None:
            r = torch.load(os.path.join(self.shard_dir, f"{pn}.pt"),
                           map_location="cpu", weights_only=True)
        else:
            r = self.pd[pn]
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray): l = torch.FloatTensor(l)
        return r, l, torch.tensor(self.t[idx], dtype=torch.float32)


class LayerDS(Dataset):
    """All-layer mean-pooled + ligand."""
    def __init__(self, prot_names, lig_names, targets, layer_dict, lig_dict, num_layers):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.lyd, self.ld, self.nl = layer_dict, lig_dict, num_layers
    def __len__(self): return len(self.t)
    def __getitem__(self, idx):
        layers = torch.stack([self.lyd[self.pn[idx]][l] for l in range(self.nl)])
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray): l = torch.FloatTensor(l)
        return layers, l, torch.tensor(self.t[idx], dtype=torch.float32)


class LayerResidueDS(Dataset):
    """All-layer + per-residue + ligand. Residue source can be a dict or
    a sharded directory path (one .pt per protein) — extract_esm2_embeddings
    returns the latter for per-residue caches.
    """
    def __init__(self, prot_names, lig_names, targets,
                 layer_dict, res_src, lig_dict, num_layers):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.lyd, self.ld, self.nl = layer_dict, lig_dict, num_layers
        if isinstance(res_src, str) and os.path.isdir(res_src):
            self.shard_dir = res_src
            self.rd = None
        else:
            self.shard_dir = None
            self.rd = res_src
    def __len__(self): return len(self.t)
    def __getitem__(self, idx):
        layers = torch.stack([self.lyd[self.pn[idx]][l] for l in range(self.nl)])
        if self.shard_dir is not None:
            r = torch.load(os.path.join(self.shard_dir, f"{self.pn[idx]}.pt"),
                           map_location="cpu", weights_only=True)
        else:
            r = self.rd[self.pn[idx]]
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray): l = torch.FloatTensor(l)
        return layers, r, l, torch.tensor(self.t[idx], dtype=torch.float32)


def residue_collate_fn(batch):
    residues, ligands, targets = zip(*batch)
    lengths = [r.shape[0] for r in residues]
    max_len = max(lengths)
    d = residues[0].shape[1]
    padded = torch.zeros(len(batch), max_len, d)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (r, l) in enumerate(zip(residues, lengths)):
        padded[i, :l] = r; mask[i, :l] = True
    return padded, torch.stack(ligands), mask, torch.stack(targets)


def layer_residue_collate_fn(batch):
    layers, residues, ligands, targets = zip(*batch)
    lengths = [r.shape[0] for r in residues]
    max_len = max(lengths)
    d = residues[0].shape[1]
    padded = torch.zeros(len(batch), max_len, d)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (r, l) in enumerate(zip(residues, lengths)):
        padded[i, :l] = r; mask[i, :l] = True
    return (torch.stack(layers), padded, torch.stack(ligands),
            mask, torch.stack(targets))


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def _mlp_head(input_dim, hidden_dims=(512, 256), dropout=0.1):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.net = _mlp_head(input_dim, hidden_dims, dropout)
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── 4 Attention Modules ──────────────────────────────────────────────────────

class ProteinOnlyAttn(nn.Module):
    def __init__(self, embed_dim, ligand_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)
    def forward(self, residues, ligand, mask):
        s = self.attn(residues).squeeze(-1)
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), residues).squeeze(1), w


class CrossAttn(nn.Module):
    def __init__(self, embed_dim, ligand_dim, attn_dim=256):
        super().__init__()
        self.q = nn.Linear(ligand_dim, attn_dim)
        self.k = nn.Linear(embed_dim, attn_dim)
        self.scale = attn_dim ** 0.5
    def forward(self, residues, ligand, mask):
        s = torch.bmm(self.q(ligand).unsqueeze(1),
                       self.k(residues).transpose(1, 2)).squeeze(1) / self.scale
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), residues).squeeze(1), w


class SelfAttnPool(nn.Module):
    def __init__(self, embed_dim, ligand_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                         batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.Linear(embed_dim, 1)
    def forward(self, residues, ligand, mask):
        kpm = ~mask if mask is not None else None
        ctx, _ = self.sa(residues, residues, residues, key_padding_mask=kpm)
        ctx = self.norm(residues + ctx)
        s = self.pool(ctx).squeeze(-1)
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), ctx).squeeze(1), w


class SelfCrossAttn(nn.Module):
    def __init__(self, embed_dim, ligand_dim, attn_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                         batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(ligand_dim, attn_dim)
        self.k = nn.Linear(embed_dim, attn_dim)
        self.scale = attn_dim ** 0.5
    def forward(self, residues, ligand, mask):
        kpm = ~mask if mask is not None else None
        ctx, _ = self.sa(residues, residues, residues, key_padding_mask=kpm)
        ctx = self.norm(residues + ctx)
        s = torch.bmm(self.q(ligand).unsqueeze(1),
                       self.k(ctx).transpose(1, 2)).squeeze(1) / self.scale
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), ctx).squeeze(1), w


class ProtQuerySelfCrossAttn(nn.Module):
    """Self-attend protein, then protein queries a K-token expansion of the
    ligand, then attention-pool the ligand-conditioned residues.

    Flow:
      residues --SelfAttn--> ctx                          (B, L, d)
      ligand   --Linear+reshape--> lig_tokens             (B, K, d)
      ctx      --CrossAttn(Q=ctx, KV=lig_tokens)--> ctx'  (B, L, d)
      ctx'     --Linear+softmax(pool)--> per-residue w    (B, L)
      output   = Σ w_i · ctx'_i                           (B, d)
    The returned attention weights `w` are over protein residues, so the
    pocket-overlap interpretability pipeline continues to work.
    """
    def __init__(self, embed_dim, ligand_dim, n_heads=4, n_lig_tokens=8,
                 dropout=0.1):
        super().__init__()
        self.n_lig_tokens = n_lig_tokens
        self.sa = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                         batch_first=True)
        self.norm_sa = nn.LayerNorm(embed_dim)
        self.lig_expand = nn.Linear(ligand_dim, embed_dim * n_lig_tokens)
        self.ca = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                         batch_first=True)
        self.norm_ca = nn.LayerNorm(embed_dim)
        self.pool = nn.Linear(embed_dim, 1)

    def forward(self, residues, ligand, mask):
        kpm = ~mask if mask is not None else None
        sa_out, _ = self.sa(residues, residues, residues, key_padding_mask=kpm)
        ctx = self.norm_sa(residues + sa_out)

        lig_tokens = self.lig_expand(ligand).view(
            ligand.size(0), self.n_lig_tokens, -1)
        ca_out, _ = self.ca(ctx, lig_tokens, lig_tokens)
        ctx = self.norm_ca(ctx + ca_out)

        s = self.pool(ctx).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), ctx).squeeze(1), w


ATTN_METHODS = OrderedDict([
    ("prot-only",   ProteinOnlyAttn),
    ("cross",       CrossAttn),
    ("self+pool",   SelfAttnPool),
    ("self+cross",  SelfCrossAttn),
    ("protq+cross", ProtQuerySelfCrossAttn),
])


# ── 3 Settings ────────────────────────────────────────────────────────────────

class AttnMLP(nn.Module):
    """Setting A: [attn_pool(residues) || ligand] → MLP"""
    def __init__(self, attn_cls, embed_dim, ligand_dim, hidden_dims=(512, 256)):
        super().__init__()
        self.attn_module = attn_cls(embed_dim, ligand_dim)
        self.head = _mlp_head(embed_dim + ligand_dim, hidden_dims)
    def forward(self, residues, ligand, mask=None):
        pooled, self._w = self.attn_module(residues, ligand, mask)
        return self.head(torch.cat([pooled, ligand], dim=1)).squeeze(-1)


class LayerWMLP(nn.Module):
    """Setting B: [layer_weighted(layers) || ligand] → MLP"""
    def __init__(self, num_layers, embed_dim, ligand_dim, hidden_dims=(512, 256)):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(num_layers))
        self.ls = nn.Parameter(torch.ones(1))
        self.head = _mlp_head(embed_dim + ligand_dim, hidden_dims)
    def forward(self, layer_embs, ligand):
        w = torch.softmax(self.lw, dim=0)
        prot = self.ls * torch.einsum("bld,l->bd", layer_embs, w)
        return self.head(torch.cat([prot, ligand], dim=1)).squeeze(-1)


class LayerWAttnMLP(nn.Module):
    """Setting C: [layer_weighted || attn_pool || ligand] → MLP"""
    def __init__(self, attn_cls, num_layers, embed_dim, ligand_dim, hidden_dims=(512, 256)):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(num_layers))
        self.ls = nn.Parameter(torch.ones(1))
        self.attn_module = attn_cls(embed_dim, ligand_dim)
        self.head = _mlp_head(embed_dim * 2 + ligand_dim, hidden_dims)
    def forward(self, layer_embs, residues, ligand, mask=None):
        w = torch.softmax(self.lw, dim=0)
        prot_global = self.ls * torch.einsum("bld,l->bd", layer_embs, w)
        prot_local, self._w = self.attn_module(residues, ligand, mask)
        return self.head(torch.cat([prot_global, prot_local, ligand], dim=1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_loop(model, train_loader, val_loader, forward_fn,
               lr=1e-3, epochs=200, patience=20,
               ckpt_path=None, log_every=1):
    """Generic training loop with optional checkpointing and per-epoch logging.

    forward_fn(model, batch) -> (pred, target).

    If `ckpt_path` is provided:
      - On entry, if the file exists, load weights and skip training entirely.
      - Otherwise, save the best state to that path after training finishes.
      The full state_dict is what's saved, so this restores the model exactly.

    `log_every` controls per-epoch progress logging. Default 1 = every epoch.
    Set to 0 to disable. Set to N>1 to print every Nth epoch only.
    """
    import time
    model = model.to(DEVICE)

    # ── Resume from checkpoint if available ──────────────────────────────────
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint {ckpt_path} — skipping training.")
        return model.to(DEVICE).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    if log_every:
        print(f"  Training (epochs<={epochs}, patience={patience}"
              + (f", ckpt={ckpt_path}" if ckpt_path else "")
              + ")")

    for epoch in range(epochs):
        ep_start = time.time()
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
        train_loss = tl / max(tn, 1)

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred, target = forward_fn(model, batch)
                vl += criterion(pred, target).item() * len(target)
                vn += len(target)
        val_loss = vl / vn
        scheduler.step(val_loss)

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if log_every and (epoch % log_every == 0 or improved):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    [ep {epoch+1:3d}] train={train_loss:.4f} "
                  f"val={val_loss:.4f} lr={cur_lr:.1e} "
                  f"best={best_loss:.4f} wait={wait}/{patience} "
                  f"({time.time() - ep_start:.1f}s)"
                  + (" *" if improved else ""), flush=True)

        if wait >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  Trained {epoch+1} ep, best val={best_loss:.4f}")

    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.save(best_state, ckpt_path)
        print(f"  Saved checkpoint {ckpt_path}")

    return model.to(DEVICE).eval()


# Forward functions for each setting
def fwd_concat(model, batch):
    x, y = batch
    return model(x.to(DEVICE)), y.to(DEVICE)

def fwd_residue(model, batch):
    r, l, m, y = batch
    return model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE)), y.to(DEVICE)

def fwd_layer(model, batch):
    ly, lg, y = batch
    return model(ly.to(DEVICE), lg.to(DEVICE)), y.to(DEVICE)

def fwd_layer_residue(model, batch):
    ly, res, lg, mask, y = batch
    return model(ly.to(DEVICE), res.to(DEVICE),
                 lg.to(DEVICE), mask.to(DEVICE)), y.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# PDBbind pocket parsing (for interpretability)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_protein_residues(pdb_path):
    """Parse protein PDB → list of (chain, resseq, icode, aa) and sequence."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    residues, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL") and residues:
                break
            if not line.startswith("ATOM"):
                continue
            chain, resseq, icode = line[21], line[22:26].strip(), line[26].strip()
            rn = line[17:20].strip()
            key = (chain, resseq, icode)
            if key not in seen and rn in protein_letters_3to1:
                seen.add(key)
                residues.append((chain, resseq, icode, protein_letters_3to1[rn]))
    return residues, "".join(r[3] for r in residues)


def get_pocket_mask(pdb_code):
    """Binary mask: True if residue is in binding pocket (from *_pocket.pdb)."""
    prot_pdb = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_protein.pdb")
    pocket_pdb = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_pocket.pdb")
    if not os.path.exists(prot_pdb) or not os.path.exists(pocket_pdb):
        return None, None
    residues, seq = parse_protein_residues(prot_pdb)
    pocket_set = set()
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                pocket_set.add((line[21], line[22:26].strip(), line[26].strip()))
    mask = np.array([(r[0], r[1], r[2]) in pocket_set for r in residues])
    return seq, mask
