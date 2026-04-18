"""
Evaluate our models on PDBbind v2016 Core Set (Test2016_290).

Training: jglaser/binding_affinity (100k samples)
Testing:  PDBbind v2016 core set (290 complexes)

Models:
  1. MLP + mean pooling (ESM-2 650M + Morgan)
  2. MLP + attention pooling (ESM-2 650M + Morgan)

BAPULM reference numbers (from their paper, arXiv:2411.04150):
  Test2016_290: r=0.914, RMSE=0.898
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/ssd_scratch/akr"
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
PDBBIND_DIR = os.path.join(SCRATCH, "data", "pdbbind_v2016", "v2016")
os.makedirs(EMBED_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = os.path.join(SCRATCH, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import concordance_index
from ablation_study import (
    ESM_CONFIGS, load_esm_model, MLPProbe, AttentionPooledMLP, residue_collate_fn,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jglaser(num_samples=100000):
    from datasets import load_dataset
    hf_cache = os.path.join(SCRATCH, "hf_cache")
    print(f"Loading {num_samples} samples from jglaser/binding_affinity...")
    ds = load_dataset("jglaser/binding_affinity",
                      split=f"train[:{num_samples}]", cache_dir=hf_cache)
    sequences = ds["seq"]
    smiles_list = ds["smiles_can"]
    targets = np.array(ds["neg_log10_affinity_M"], dtype=np.float32)

    valid = ~np.isnan(targets)
    sequences = [s for s, v in zip(sequences, valid) if v]
    smiles_list = [s for s, v in zip(smiles_list, valid) if v]
    targets = targets[valid]
    print(f"Loaded {len(targets)} valid samples")
    return sequences, smiles_list, targets


def load_pdbbind_test():
    from rdkit import Chem
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    index_file = os.path.join(PDBBIND_DIR, "index", "INDEX_core_data.2016")
    pdb_codes, affinities = [], []
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdb_codes.append(parts[0])
            affinities.append(float(parts[3]))

    parser = PDBParser(QUIET=True)
    sequences, smiles_list, valid_codes, valid_affinities = [], [], [], []

    for pdb_code, aff in zip(pdb_codes, affinities):
        pdb_file = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_protein.pdb")
        sdf_file = os.path.join(PDBBIND_DIR, pdb_code, f"{pdb_code}_ligand.sdf")
        if not os.path.exists(pdb_file) or not os.path.exists(sdf_file):
            continue
        try:
            structure = parser.get_structure(pdb_code, pdb_file)
            seq = ""
            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname().strip()
                        if resname in protein_letters_3to1:
                            seq += protein_letters_3to1[resname]
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
        sequences.append(seq)
        smiles_list.append(smi)
        valid_codes.append(pdb_code)
        valid_affinities.append(aff)

    print(f"Loaded {len(valid_codes)} / {len(pdb_codes)} PDBbind complexes")
    return valid_codes, sequences, smiles_list, np.array(valid_affinities)


# ── Embedding extraction ─────────────────────────────────────────────────────

def extract_esm2_mean(sequences, names, cache_name, label="650M"):
    cfg = ESM_CONFIGS[label]
    cache_path = os.path.join(EMBED_DIR, f"esm2_{label}_mean_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached ESM-2 {label} mean-pooled ({cache_name})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    model, alphabet = load_esm_model(cfg["hub"])
    batch_converter = alphabet.get_batch_converter()
    num_layers = cfg["layers"]

    unique = list(set(zip(names, sequences)))
    embed_dict = {}
    print(f"Extracting ESM-2 {label} mean-pooled for {len(unique)} proteins...")
    batch_size = 4
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i+batch_size]
        data = [(n, s[:1022]) for n, s in batch]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = model(tokens, repr_layers=[num_layers])
        for j, (name, seq) in enumerate(batch):
            sl = min(len(seq), 1022)
            embed_dict[name] = results["representations"][num_layers][
                j, 1:sl+1, :].mean(dim=0).cpu()
        if (i // batch_size + 1) % 200 == 0 or i == 0:
            print(f"  [{min(i+batch_size, len(unique))}/{len(unique)}]")

    del model; torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    return embed_dict


def extract_esm2_residue(sequences, names, cache_name, label="650M"):
    cfg = ESM_CONFIGS[label]
    cache_path = os.path.join(EMBED_DIR, f"esm2_{label}_residue_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached ESM-2 {label} residue ({cache_name})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    model, alphabet = load_esm_model(cfg["hub"])
    batch_converter = alphabet.get_batch_converter()
    num_layers = cfg["layers"]

    unique = list(set(zip(names, sequences)))
    embed_dict = {}
    print(f"Extracting ESM-2 {label} per-residue for {len(unique)} proteins...")
    for idx, (name, seq) in enumerate(unique):
        seq = seq[:1022]
        _, _, tokens = batch_converter([(name, seq)])
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = model(tokens, repr_layers=[num_layers])
        embed_dict[name] = results["representations"][num_layers][
            0, 1:len(seq)+1, :].cpu()
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique)}]")

    del model; torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    return embed_dict


def compute_morgan(smiles_list, names, n_bits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_dict = {}
    for name, smi in zip(names, smiles_list):
        if name in fp_dict:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp_dict[name] = np.zeros(n_bits, dtype=np.float64)
        else:
            fp_dict[name] = gen.GetFingerprintAsNumPy(mol).astype(np.float64)
    return fp_dict


# ── Datasets ──────────────────────────────────────────────────────────────────

class ConcatDataset(Dataset):
    def __init__(self, prot_names, lig_names, targets, prot_dict, lig_dict):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.pd, self.ld = prot_dict, lig_dict

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        p = self.pd[self.pn[idx]]
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray):
            l = torch.FloatTensor(l)
        return torch.cat([p, l]), torch.tensor(self.t[idx], dtype=torch.float32)


class ResidueDataset(Dataset):
    def __init__(self, prot_names, lig_names, targets, prot_dict, lig_dict):
        self.pn, self.ln, self.t = prot_names, lig_names, targets
        self.pd, self.ld = prot_dict, lig_dict

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        r = self.pd[self.pn[idx]]
        l = self.ld[self.ln[idx]]
        if isinstance(l, np.ndarray):
            l = torch.FloatTensor(l)
        return r, l, torch.tensor(self.t[idx], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp(model, train_loader, val_loader, epochs=200, patience=25, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        vl = sum(criterion(model(x.to(DEVICE)), y.to(DEVICE)).item() * len(y)
                 for x, y in val_loader)
        val_loss = vl / len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Val MSE: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model.to(DEVICE).eval()


def train_attn(model, train_loader, val_loader, epochs=200, patience=25, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for r, l, m, y in train_loader:
            r, l, m, y = r.to(DEVICE), l.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(r, l, m), y).backward()
            optimizer.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for r, l, m, y in val_loader:
                r, l, m, y = r.to(DEVICE), l.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                vl += criterion(model(r, l, m), y).item() * len(y)
        val_loss = vl / len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Val MSE: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model.to(DEVICE).eval()


def evaluate(y_true, y_pred, name=""):
    r, _ = stats.pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ci = concordance_index(y_true, y_pred)
    print(f"  {name}: r={r:.4f} | RMSE={rmse:.4f} | CI={ci:.4f}")
    return {"name": name, "r": r, "rmse": rmse, "ci": ci}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("PDBbind EVALUATION: Our Models")
    print("Training: jglaser 100k | Testing: Test2016_290")
    print("=" * 80)

    # Load training data
    train_seqs, train_smiles, train_targets = load_jglaser(100000)
    target_mean, target_std = train_targets.mean(), train_targets.std()
    train_targets_std = (train_targets - target_mean) / target_std

    seq_to_name = {}
    for s in train_seqs:
        if s not in seq_to_name:
            seq_to_name[s] = f"prot_{len(seq_to_name)}"
    smi_to_name = {}
    for s in train_smiles:
        if s not in smi_to_name:
            smi_to_name[s] = f"lig_{len(smi_to_name)}"
    train_pn = [seq_to_name[s] for s in train_seqs]
    train_ln = [smi_to_name[s] for s in train_smiles]

    tr_idx, va_idx = train_test_split(
        np.arange(len(train_targets)), test_size=0.1, random_state=42)

    # Load test data
    pdb_codes, test_seqs, test_smiles, test_targets = load_pdbbind_test()

    # Extract embeddings
    print("\n" + "=" * 80)
    print("Extracting embeddings")
    print("=" * 80)

    train_esm_mean = extract_esm2_mean(
        list(seq_to_name.keys()), list(seq_to_name.values()), "jglaser100k")
    train_esm_res = extract_esm2_residue(
        list(seq_to_name.keys()), list(seq_to_name.values()), "jglaser100k")
    train_morgan = compute_morgan(list(smi_to_name.keys()), list(smi_to_name.values()))

    test_esm_mean = extract_esm2_mean(test_seqs, pdb_codes, "pdbbind_test")
    test_esm_res = extract_esm2_residue(test_seqs, pdb_codes, "pdbbind_test")
    test_morgan = compute_morgan(test_smiles, pdb_codes)

    cfg = ESM_CONFIGS["650M"]
    all_results = []
    BS = 512

    # ── Model 1: MLP + mean pooling ──
    print("\n" + "=" * 80)
    print("Model 1: MLP + mean pool")
    print("=" * 80)

    tr_ds = ConcatDataset([train_pn[i] for i in tr_idx],
                          [train_ln[i] for i in tr_idx],
                          train_targets_std[tr_idx], train_esm_mean, train_morgan)
    va_ds = ConcatDataset([train_pn[i] for i in va_idx],
                          [train_ln[i] for i in va_idx],
                          train_targets_std[va_idx], train_esm_mean, train_morgan)
    model = train_mlp(
        MLPProbe(cfg["dim"] + 2048, (512, 256)),
        DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=4),
        DataLoader(va_ds, batch_size=BS, num_workers=4))

    te_ds = ConcatDataset(pdb_codes, pdb_codes, test_targets,
                          test_esm_mean, test_morgan)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(te_ds, batch_size=256):
            preds.append(model(x.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * target_std + target_mean
    all_results.append(evaluate(test_targets, y_pred, "MLP+mean (ESM2-650M+Morgan)"))
    del model; torch.cuda.empty_cache()

    # ── Model 2: MLP + attention pooling ──
    print("\n" + "=" * 80)
    print("Model 2: MLP + attention pool")
    print("=" * 80)

    tr_res = ResidueDataset([train_pn[i] for i in tr_idx],
                            [train_ln[i] for i in tr_idx],
                            train_targets_std[tr_idx], train_esm_res, train_morgan)
    va_res = ResidueDataset([train_pn[i] for i in va_idx],
                            [train_ln[i] for i in va_idx],
                            train_targets_std[va_idx], train_esm_res, train_morgan)
    model = train_attn(
        AttentionPooledMLP(cfg["dim"], 2048, (512, 256)),
        DataLoader(tr_res, batch_size=BS, shuffle=True,
                   collate_fn=residue_collate_fn, num_workers=0),
        DataLoader(va_res, batch_size=BS,
                   collate_fn=residue_collate_fn, num_workers=0))

    te_res = ResidueDataset(pdb_codes, pdb_codes, test_targets,
                            test_esm_res, test_morgan)
    preds = []
    with torch.no_grad():
        for r, l, m, _ in DataLoader(te_res, batch_size=256,
                                     collate_fn=residue_collate_fn):
            preds.append(model(r.to(DEVICE), l.to(DEVICE), m.to(DEVICE)).cpu())
    y_pred = torch.cat(preds).numpy() * target_std + target_mean
    all_results.append(evaluate(test_targets, y_pred, "MLP+attn (ESM2-650M+Morgan)"))
    del model; torch.cuda.empty_cache()

    # ── Results ──
    print("\n" + "=" * 80)
    print("RESULTS: PDBbind Test2016_290")
    print("=" * 80)
    print(f"\n{'Model':<40s} | {'r':>6s} | {'RMSE':>6s} | {'CI':>6s}")
    print("-" * 65)
    for r in all_results:
        print(f"  {r['name']:<38s} | {r['r']:>6.4f} | {r['rmse']:>6.4f} | "
              f"{r['ci']:>6.4f}")
    print(f"  {'BAPULM (paper, arXiv:2411.04150)':<38s} | {'0.914':>6s} | "
          f"{'0.898':>6s} | {'  -  ':>6s}")


if __name__ == "__main__":
    main()
