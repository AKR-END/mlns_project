"""
Train ESM-2 650M + Morgan Fingerprint MLP on binding affinity prediction.

Dataset: All 1.84M samples from jglaser/binding_affinity (HuggingFace)
Model: Concat(ESM-2 650M mean-pooled, Morgan FP 2048) -> MLP [512, 256] w/ BatchNorm -> 1
Loss: MSE (on standardized targets)
Split: 90:10 train/val
Eval: Test2016_290 (external test set)
Metrics: Pearson R, RMSE, MSE
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import mean_squared_error
import math

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH = "/ssd_scratch/akr"
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
EMBED_DIR = os.path.join(SCRATCH, "embeddings", "binding_affinity")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── MLP Model ─────────────────────────────────────────────────────────────────
class AffinityMLP(nn.Module):
    def __init__(self, protein_dim=1280, ligand_dim=2048, hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = protein_dim + ligand_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_hf_dataset(num_samples=None):
    """Load from jglaser/binding_affinity. If num_samples is None, load all.
    Columns: seq, smiles_can, neg_log10_affinity_M, affinity, affinity_uM, smiles
    """
    from datasets import load_dataset
    if num_samples is None:
        print("Loading ALL samples from jglaser/binding_affinity...")
        ds = load_dataset("jglaser/binding_affinity", split="train",
                           cache_dir=os.path.join(SCRATCH, "hf_cache"))
    else:
        print(f"Loading first {num_samples} samples from jglaser/binding_affinity...")
        ds = load_dataset("jglaser/binding_affinity", split=f"train[:{num_samples}]",
                           cache_dir=os.path.join(SCRATCH, "hf_cache"))
    print(f"Loaded {len(ds)} samples. Columns: {ds.column_names}")
    return ds


def load_pdbbind_test(data_dir):
    """Load PDBbind v2016 core set (Test2016_290).

    Parses:
    - Binding affinities from INDEX_core_data.2016
    - Protein sequences from *_protein.pdb (extracts from ATOM records)
    - Ligand SMILES from *_ligand.sdf (via RDKit)
    """
    import pandas as pd
    from rdkit import Chem
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    index_file = os.path.join(data_dir, "index", "INDEX_core_data.2016")

    # Parse index: PDB code -> -logKd/Ki
    pdb_codes = []
    affinities = []
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdb_code = parts[0]
            neg_log_kd_ki = float(parts[3])
            pdb_codes.append(pdb_code)
            affinities.append(neg_log_kd_ki)

    print(f"Parsed {len(pdb_codes)} entries from index")

    # Extract protein sequences from PDB files
    parser = PDBParser(QUIET=True)
    sequences = []
    smiles_list = []
    valid_codes = []
    valid_affinities = []

    for i, (pdb_code, aff) in enumerate(zip(pdb_codes, affinities)):
        pdb_dir = os.path.join(data_dir, pdb_code)
        pdb_file = os.path.join(pdb_dir, f"{pdb_code}_protein.pdb")
        sdf_file = os.path.join(pdb_dir, f"{pdb_code}_ligand.sdf")

        if not os.path.exists(pdb_file) or not os.path.exists(sdf_file):
            print(f"  Skipping {pdb_code}: missing files")
            continue

        # Extract sequence from PDB
        try:
            structure = parser.get_structure(pdb_code, pdb_file)
            seq = ""
            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname().strip()
                        if resname in protein_letters_3to1:
                            seq += protein_letters_3to1[resname]
                break  # first model only
        except Exception as e:
            print(f"  Skipping {pdb_code}: PDB parse error: {e}")
            continue

        if len(seq) == 0:
            print(f"  Skipping {pdb_code}: empty sequence")
            continue

        # Extract SMILES from SDF (try sanitized first, fallback to unsanitized)
        try:
            supplier = Chem.SDMolSupplier(sdf_file, sanitize=True)
            mol = next(iter(supplier))
            if mol is None:
                supplier = Chem.SDMolSupplier(sdf_file, sanitize=False)
                mol = next(iter(supplier))
            if mol is None:
                print(f"  Skipping {pdb_code}: RDKit failed to read ligand")
                continue
            smi = Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"  Skipping {pdb_code}: SDF parse error: {e}")
            continue

        sequences.append(seq)
        smiles_list.append(smi)
        valid_codes.append(pdb_code)
        valid_affinities.append(aff)

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(pdb_codes)}]")

    print(f"Successfully loaded {len(valid_codes)} / {len(pdb_codes)} complexes")
    return valid_codes, sequences, smiles_list, np.array(valid_affinities, dtype=np.float32)


# ── Feature Extraction ────────────────────────────────────────────────────────
def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for a list of SMILES strings."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = {}
    failed = 0
    for smi in smiles_list:
        if smi in fps:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            fps[smi] = np.zeros(n_bits, dtype=np.float32)
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        fps[smi] = fp.astype(np.float32)
    if failed > 0:
        print(f"  Warning: {failed} SMILES failed to parse")
    return fps


def extract_esm2_embeddings(sequences, names, cache_name="binding_affinity"):
    """Extract mean-pooled ESM-2 650M last-layer embeddings."""
    cache_path = os.path.join(EMBED_DIR, f"esm2_650M_{cache_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    import esm

    model_ckpt = os.path.join(CACHE_DIR, "hub", "checkpoints", "esm2_t33_650M_UR50D.pt")
    if os.path.exists(model_ckpt):
        print(f"Loading 650M model from {model_ckpt}...")
        model_data = torch.load(model_ckpt, map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            "esm2_t33_650M_UR50D", model_data, None)
    else:
        print("Downloading ESM-2 650M model...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    embed_dict = {}
    # Process in batches for efficiency
    batch_size = 4
    unique_seqs = list(set(zip(names, sequences)))
    print(f"Extracting embeddings for {len(unique_seqs)} unique proteins...")

    for i in range(0, len(unique_seqs), batch_size):
        batch = unique_seqs[i:i + batch_size]
        data = [(n, s[:1022].upper()) for n, s in batch]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers])

        for j, (name, seq) in enumerate(batch):
            seq_len = min(len(seq), 1022)
            rep = results["representations"][num_layers][j, 1:seq_len + 1, :]
            embed_dict[name] = rep.mean(dim=0).cpu()

        if (i // batch_size + 1) % 100 == 0 or i == 0:
            print(f"  [{i + len(batch)}/{len(unique_seqs)}]")

    del model
    torch.cuda.empty_cache()

    torch.save(embed_dict, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return embed_dict


# ── Dataset Class ─────────────────────────────────────────────────────────────
class AffinityDataset(Dataset):
    """Lazy-loading dataset: looks up embeddings/fps from dicts on-the-fly to avoid OOM."""
    def __init__(self, prot_names, smiles_list, targets, embed_dict, fp_dict):
        self.prot_names = prot_names
        self.smiles_list = smiles_list
        self.targets = targets
        self.embed_dict = embed_dict
        self.fp_dict = fp_dict

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        prot_emb = self.embed_dict[self.prot_names[idx]]
        lig_fp = torch.FloatTensor(self.fp_dict[self.smiles_list[idx]])
        feat = torch.cat([prot_emb, lig_fp])
        return feat, torch.tensor(self.targets[idx], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, lr=1e-3, epochs=100, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                val_loss_sum += loss.item() * len(yb)
                val_n += len(yb)

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best at {epoch+1-wait})")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE).eval()
    print(f"  Best val MSE: {best_val_loss:.4f}")
    return model


def evaluate(model, data_loader, name="Test", target_mean=0.0, target_std=1.0):
    """Evaluate model. Predictions are in standardized space; denormalize before metrics."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    y_pred_std = np.concatenate(all_preds)
    y_true_std = np.concatenate(all_targets)

    # Denormalize back to original pK scale
    y_pred = y_pred_std * target_std + target_mean
    y_true = y_true_std * target_std + target_mean

    r, p_val = stats.pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    print(f"\n  {name} Results:")
    print(f"    Pearson R : {r:.4f} (p={p_val:.2e})")
    print(f"    RMSE      : {rmse:.4f}")
    print(f"    MSE       : {mse:.4f}")
    return {"pearson_r": r, "rmse": rmse, "mse": mse}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to PDBbind v2016 directory (contains index/ and PDB subdirs)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load training data ──
    print("=" * 70)
    print("Step 1: Loading training data (ALL from jglaser/binding_affinity)")
    print("=" * 70)
    ds = load_hf_dataset(None)

    # Columns: seq, smiles_can, neg_log10_affinity_M
    sequences = ds['seq']
    smiles_list = ds['smiles_can']
    targets = np.array(ds['neg_log10_affinity_M'], dtype=np.float32)

    # Filter out any NaN targets
    valid_mask = ~np.isnan(targets)
    sequences = [s for s, v in zip(sequences, valid_mask) if v]
    smiles_list = [s for s, v in zip(smiles_list, valid_mask) if v]
    targets = targets[valid_mask]
    print(f"Valid samples after NaN filter: {len(targets)}")

    # Standardize targets (zero-mean, unit-variance)
    target_mean = targets.mean()
    target_std = targets.std()
    targets = (targets - target_mean) / target_std
    print(f"Target standardization: mean={target_mean:.4f}, std={target_std:.4f}")

    # Create unique protein names (collision-free)
    seq_to_name = {}
    counter = 0
    for seq in sequences:
        if seq not in seq_to_name:
            seq_to_name[seq] = f"prot_{counter}"
            counter += 1
    prot_names = [seq_to_name[s] for s in sequences]

    # ── Compute features ──
    print("\n" + "=" * 70)
    print("Step 2: Computing ligand fingerprints")
    print("=" * 70)
    unique_smiles = list(set(smiles_list))
    print(f"Unique SMILES: {len(unique_smiles)}")
    fp_dict = compute_morgan_fingerprints(unique_smiles)

    print("\n" + "=" * 70)
    print("Step 3: Extracting ESM-2 650M embeddings (training proteins)")
    print("=" * 70)
    unique_names = list(seq_to_name.values())
    unique_seqs = list(seq_to_name.keys())
    print(f"Unique proteins: {len(unique_names)}")
    embed_dict = extract_esm2_embeddings(unique_seqs, unique_names, cache_name="train_full")

    # ── 90:10 train/val split ──
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=args.seed)
    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}")

    # Build lazy-loading datasets (avoids materializing full feature matrices)
    train_names = [prot_names[i] for i in train_idx]
    train_smiles = [smiles_list[i] for i in train_idx]
    val_names = [prot_names[i] for i in val_idx]
    val_smiles = [smiles_list[i] for i in val_idx]

    train_ds = AffinityDataset(train_names, train_smiles, targets[train_idx], embed_dict, fp_dict)
    val_ds = AffinityDataset(val_names, val_smiles, targets[val_idx], embed_dict, fp_dict)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Train ──
    print("\n" + "=" * 70)
    print("Step 4: Training MLP [512, 256]")
    print("=" * 70)
    model = AffinityMLP(protein_dim=1280, ligand_dim=2048,
                         hidden_dims=(512, 256), dropout=0.1).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train(model, train_loader, val_loader,
                  lr=args.lr, epochs=args.epochs, patience=25)

    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("Step 5: Evaluation on validation set")
    print("=" * 70)
    val_results = evaluate(model, val_loader, name="Validation",
                           target_mean=target_mean, target_std=target_std)

    # ── Load and evaluate on Test2016_290 ──
    print("\n" + "=" * 70)
    print("Step 6: Evaluation on Test2016_290 (PDBbind v2016 core set)")
    print("=" * 70)
    test_codes, test_sequences, test_smiles, test_targets = load_pdbbind_test(args.test_dir)

    # Compute test features
    print("\nComputing test ligand fingerprints...")
    test_fp_dict = compute_morgan_fingerprints(test_smiles)

    print("Extracting test protein embeddings...")
    test_prot_names = test_codes  # use PDB codes as names
    test_unique_pairs = list(set(zip(test_prot_names, test_sequences)))
    test_embed_dict = extract_esm2_embeddings(
        [s for _, s in test_unique_pairs],
        [n for n, _ in test_unique_pairs],
        cache_name="test2016_290")

    # Standardize test targets using training stats
    test_targets_std = (test_targets - target_mean) / target_std
    test_ds = AffinityDataset(test_prot_names, test_smiles, test_targets_std,
                               test_embed_dict, test_fp_dict)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_results = evaluate(model, test_loader, name="Test2016_290",
                            target_mean=target_mean, target_std=target_std)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20s} | {'Pearson R':>10s} | {'RMSE':>10s} | {'MSE':>10s}")
    print("-" * 60)
    print(f"{'Validation':<20s} | {val_results['pearson_r']:>10.4f} | {val_results['rmse']:>10.4f} | {val_results['mse']:>10.4f}")
    print(f"{'Test2016_290':<20s} | {test_results['pearson_r']:>10.4f} | {test_results['rmse']:>10.4f} | {test_results['mse']:>10.4f}")

    # Save model
    save_path = os.path.join(SCRATCH, "binding_affinity_mlp.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
