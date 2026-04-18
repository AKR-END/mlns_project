"""
Ablation study for protein-ligand binding affinity prediction.

Systematically evaluates how pooling strategy, model scale, ligand
representation, and probe architecture affect performance.

Experiment matrix (Davis dataset, 64/16/20 split):
  Protein encoder: ESM-2 {8M, 35M, 650M}
  Pooling:         {mean, max, attention-weighted}
  Ligand repr:     {Morgan FP (2048-d), ChemBERTa (384-d)}
  Probe:           {Ridge, MLP [512,256], LayerW+MLP [512,256]}

Notes:
  - Ridge only with mean/max (not learnable, no attention)
  - Attention pooling only with MLP
  - LayerW+MLP uses all-layer mean-pooled embeddings
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH = "/ssd_scratch/akr"
DATA_DIR = os.path.join(SCRATCH, "data", "davis")
CACHE_DIR = os.path.join(SCRATCH, "model_cache")
EMBED_DIR = os.path.join(SCRATCH, "embeddings")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = os.path.join(SCRATCH, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

sys.path.insert(0, os.path.dirname(__file__))
from feasibility_test import (
    download_davis, load_davis, compute_ligand_fingerprints, concordance_index,
)

# ── ESM-2 model configs ──────────────────────────────────────────────────────
ESM_CONFIGS = OrderedDict([
    ("8M",   {"hub": "esm2_t6_8M_UR50D",   "layers": 6,  "dim": 320}),
    ("35M",  {"hub": "esm2_t12_35M_UR50D",  "layers": 12, "dim": 480}),
    ("650M", {"hub": "esm2_t33_650M_UR50D", "layers": 33, "dim": 1280}),
])


# ── ESM-2 model loading ──────────────────────────────────────────────────────
def load_esm_model(hub_name):
    import esm
    ckpt = os.path.join(CACHE_DIR, "hub", "checkpoints", f"{hub_name}.pt")
    if os.path.exists(ckpt):
        print(f"  Loading from checkpoint: {ckpt}")
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            hub_name, data, None)
    else:
        print(f"  Downloading {hub_name} via torch hub...")
        model, alphabet = esm.pretrained.load_model_and_alphabet(hub_name)
    return model.eval().to(DEVICE), alphabet


# ── Per-residue embedding extraction (last layer) ────────────────────────────
def extract_residue_embeddings(df, label):
    """Extract per-residue last-layer ESM-2 embeddings.
    Returns: {protein_name: tensor(seq_len, embed_dim)}
    """
    cfg = ESM_CONFIGS[label]
    cache_path = os.path.join(EMBED_DIR, f"{cfg['hub']}_residue_davis.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached residue embeddings ({label})")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    model, alphabet = load_esm_model(cfg["hub"])
    batch_converter = alphabet.get_batch_converter()
    num_layers = cfg["layers"]

    unique_prots = df[["protein_name", "sequence"]].drop_duplicates()
    embed_dict = {}

    print(f"Extracting per-residue embeddings ({label}) for {len(unique_prots)} proteins...")
    for idx, (_, row) in enumerate(unique_prots.iterrows()):
        name = row["protein_name"]
        seq = row["sequence"][:1022]
        _, _, batch_tokens = batch_converter([(name, seq)])
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers])

        seq_len = len(seq)
        embed_dict[name] = results["representations"][num_layers][0, 1:seq_len+1, :].cpu()

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique_prots)}] {name} (len={seq_len})")

    del model
    torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    print(f"Saved to {cache_path}")
    return embed_dict


# ── All-layer mean-pooled extraction ─────────────────────────────────────────
def extract_all_layer_embeddings(df, label):
    """Extract mean-pooled per-layer ESM-2 embeddings.
    Returns: {protein_name: {layer_idx: tensor(embed_dim,)}}
    """
    cfg = ESM_CONFIGS[label]
    cache_candidates = [
        os.path.join(EMBED_DIR, f"{cfg['hub']}_all_layers_davis.pt"),
        os.path.join(EMBED_DIR, f"{cfg['hub']}_davis.pt"),
    ]
    for path in cache_candidates:
        if os.path.exists(path):
            print(f"Loading cached all-layer embeddings ({label}) from {path}")
            return torch.load(path, map_location="cpu", weights_only=False)

    model, alphabet = load_esm_model(cfg["hub"])
    batch_converter = alphabet.get_batch_converter()
    num_layers = cfg["layers"]

    unique_prots = df[["protein_name", "sequence"]].drop_duplicates()
    embed_dict = {}

    print(f"Extracting all-layer embeddings ({label}) for {len(unique_prots)} proteins...")
    for idx, (_, row) in enumerate(unique_prots.iterrows()):
        name = row["protein_name"]
        seq = row["sequence"][:1022]
        _, _, batch_tokens = batch_converter([(name, seq)])
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(num_layers + 1)))

        seq_len = len(seq)
        layer_embeds = {}
        for l in range(num_layers + 1):
            rep = results["representations"][l][0, 1:seq_len+1, :]
            layer_embeds[l] = rep.mean(dim=0).cpu()
        embed_dict[name] = layer_embeds

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unique_prots)}] {name} (len={seq_len})")

    del model
    torch.cuda.empty_cache()
    save_path = os.path.join(EMBED_DIR, f"{cfg['hub']}_all_layers_davis.pt")
    torch.save(embed_dict, save_path)
    print(f"Saved to {save_path}")
    return embed_dict


# ── Learned ligand embedding extraction (ChemBERTa) ──────────────────────────
def extract_learned_ligand_embeddings(df):
    """Extract ChemBERTa embeddings for all unique ligands.
    Uses DeepChem/ChemBERTa-77M-MLM (RoBERTa trained on 77M SMILES).
    Returns: {ligand_name: np.array(768,)}
    """
    cache_path = os.path.join(EMBED_DIR, "chemberta_davis.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached ChemBERTa embeddings")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from transformers import AutoModel, AutoTokenizer

    print("Loading ChemBERTa-77M-MLM model...")
    hf_cache = os.path.join(SCRATCH, "hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", cache_dir=hf_cache)
    model = AutoModel.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", cache_dir=hf_cache)
    model = model.eval().to(DEVICE)

    unique_ligands = df[["ligand_name", "smiles"]].drop_duplicates()
    embed_dict = {}
    names = unique_ligands["ligand_name"].tolist()
    smiles = unique_ligands["smiles"].tolist()

    print(f"Extracting ChemBERTa embeddings for {len(names)} ligands...")
    batch_size = 32
    for i in range(0, len(names), batch_size):
        batch_smiles = smiles[i:i+batch_size]
        batch_names = names[i:i+batch_size]

        inputs = tokenizer(batch_smiles, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embs = outputs.pooler_output.cpu().numpy()
        else:
            embs = outputs.last_hidden_state[:, 0].cpu().numpy()

        for name, emb in zip(batch_names, embs):
            embed_dict[name] = emb.astype(np.float32)

    del model
    torch.cuda.empty_cache()
    torch.save(embed_dict, cache_path)
    print(f"Saved to {cache_path}")
    return embed_dict


# ── Pooling utilities ─────────────────────────────────────────────────────────
def pool_residues(residue_dict, method="mean"):
    """Convert per-residue embeddings to fixed-size vectors.
    Returns: {protein_name: np.array(embed_dim,)}
    """
    pooled = {}
    for name, emb in residue_dict.items():
        if method == "mean":
            pooled[name] = emb.mean(dim=0).numpy()
        elif method == "max":
            pooled[name] = emb.max(dim=0)[0].numpy()
        else:
            raise ValueError(f"Unknown pooling: {method}")
    return pooled


def build_feature_matrix(df, protein_dict, ligand_dict):
    """Build [protein || ligand] feature matrix. Returns: np.array (n, d)."""
    X = []
    for _, row in df.iterrows():
        prot = protein_dict[row["protein_name"]]
        lig = ligand_dict[row["ligand_name"]]
        X.append(np.concatenate([prot, lig]))
    return np.array(X)


# ── Model definitions ─────────────────────────────────────────────────────────
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionPooledMLP(nn.Module):
    """Learned attention pooling over residues + MLP head."""
    def __init__(self, embed_dim, ligand_dim, hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)
        input_dim = embed_dim + ligand_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, residue_embs, ligand_fp, mask=None):
        attn_scores = self.attn(residue_embs).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), residue_embs).squeeze(1)
        combined = torch.cat([pooled, ligand_fp], dim=1)
        return self.head(combined).squeeze(-1)


class LayerWeightedProbe(nn.Module):
    """Learned scalar mixture over ESM-2 layers + MLP head."""
    def __init__(self, num_layers, embed_dim, ligand_dim,
                 hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        input_dim = embed_dim + ligand_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, layer_embeddings, ligand_fp):
        weights = torch.softmax(self.layer_weights, dim=0)
        protein_emb = self.layer_scale * torch.einsum("bld,l->bd",
                                                       layer_embeddings, weights)
        combined = torch.cat([protein_emb, ligand_fp], dim=1)
        return self.head(combined).squeeze(-1)


# ── Dataset classes ───────────────────────────────────────────────────────────
class ResidueDataset(Dataset):
    """Per-residue embeddings + ligand features for attention pooling."""
    def __init__(self, df, indices, residue_dict, ligand_dict):
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.residue_dict = residue_dict
        self.ligand_dict = ligand_dict

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        residues = self.residue_dict[row["protein_name"]]
        ligand = torch.FloatTensor(self.ligand_dict[row["ligand_name"]])
        target = torch.tensor(row["pKd"], dtype=torch.float32)
        return residues, ligand, target


def residue_collate_fn(batch):
    """Pad variable-length residue embeddings."""
    residues, ligands, targets = zip(*batch)
    lengths = [r.shape[0] for r in residues]
    max_len = max(lengths)
    embed_dim = residues[0].shape[1]

    padded = torch.zeros(len(batch), max_len, embed_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (r, length) in enumerate(zip(residues, lengths)):
        padded[i, :length] = r
        mask[i, :length] = True

    return padded, torch.stack(ligands), mask, torch.stack(targets)


class LayerDataset(Dataset):
    """All-layer mean-pooled embeddings + ligand features."""
    def __init__(self, df, indices, layer_dict, ligand_dict, num_layers):
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.layer_dict = layer_dict
        self.ligand_dict = ligand_dict
        self.num_layers = num_layers

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        pname = row["protein_name"]
        lname = row["ligand_name"]
        layer_stack = torch.stack(
            [self.layer_dict[pname][l] for l in range(self.num_layers)])
        ligand = torch.FloatTensor(self.ligand_dict[lname])
        target = torch.tensor(row["pKd"], dtype=torch.float32)
        return layer_stack, ligand, target


# ── Training functions ────────────────────────────────────────────────────────
def train_mlp_probe(X_train, y_train, X_val, y_val, hidden_dims=(512, 256),
                    lr=1e-3, epochs=100, batch_size=512, patience=10):
    model = MLPProbe(X_train.shape[1], hidden_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model.to(DEVICE).eval()


def train_attention_mlp(df, train_idx, val_idx, residue_dict, ligand_dict,
                        embed_dim, ligand_dim, hidden_dims=(512, 256),
                        lr=1e-3, epochs=100, batch_size=512, patience=10):
    train_ds = ResidueDataset(df, train_idx, residue_dict, ligand_dict)
    val_ds = ResidueDataset(df, val_idx, residue_dict, ligand_dict)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=residue_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=residue_collate_fn, num_workers=0)

    model = AttentionPooledMLP(embed_dim, ligand_dim, hidden_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for res_b, lig_b, mask_b, y_b in train_loader:
            res_b = res_b.to(DEVICE)
            lig_b = lig_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(res_b, lig_b, mask_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for res_b, lig_b, mask_b, y_b in val_loader:
                res_b = res_b.to(DEVICE)
                lig_b = lig_b.to(DEVICE)
                mask_b = mask_b.to(DEVICE)
                y_b = y_b.to(DEVICE)
                val_losses.append(
                    criterion(model(res_b, lig_b, mask_b), y_b).item() * len(y_b))
        val_loss = sum(val_losses) / len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model.to(DEVICE).eval()


def train_layer_weighted_probe(df, train_idx, val_idx, layer_dict, ligand_dict,
                               num_layers, embed_dim, ligand_dim,
                               hidden_dims=(512, 256), lr=1e-3, epochs=100,
                               batch_size=512, patience=10):
    train_ds = LayerDataset(df, train_idx, layer_dict, ligand_dict, num_layers)
    val_ds = LayerDataset(df, val_idx, layer_dict, ligand_dict, num_layers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = LayerWeightedProbe(num_layers, embed_dim, ligand_dim,
                               hidden_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for emb_b, lig_b, y_b in train_loader:
            emb_b, lig_b, y_b = emb_b.to(DEVICE), lig_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(emb_b, lig_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for emb_b, lig_b, y_b in val_loader:
                emb_b, lig_b, y_b = (emb_b.to(DEVICE), lig_b.to(DEVICE),
                                      y_b.to(DEVICE))
                val_losses.append(
                    criterion(model(emb_b, lig_b), y_b).item() * len(y_b))
        val_loss = sum(val_losses) / len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model.to(DEVICE).eval()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred, name=""):
    r, _ = stats.pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    return {"name": name, "r": r, "mse": mse, "r2": r2, "ci": ci}


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def predict_attention(model, df, indices, residue_dict, ligand_dict,
                      batch_size=512):
    ds = ResidueDataset(df, indices, residue_dict, ligand_dict)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=residue_collate_fn, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for res_b, lig_b, mask_b, _ in loader:
            res_b = res_b.to(DEVICE)
            lig_b = lig_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            preds.append(model(res_b, lig_b, mask_b).cpu())
    return torch.cat(preds).numpy()


def predict_layer_weighted(model, df, indices, layer_dict, ligand_dict,
                           num_layers, batch_size=512):
    ds = LayerDataset(df, indices, layer_dict, ligand_dict, num_layers)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for emb_b, lig_b, _ in loader:
            emb_b, lig_b = emb_b.to(DEVICE), lig_b.to(DEVICE)
            preds.append(model(emb_b, lig_b).cpu())
    return torch.cat(preds).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("ABLATION STUDY: Pooling × Scale × Ligand Rep × Probe")
    print("=" * 80)

    # ── Load data ──
    download_davis()
    df = load_davis()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 3-way split: 64% train / 16% val / 20% test
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=42)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    y_all = df["pKd"].values
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    y_train_val = np.concatenate([y_train, y_val])

    # ── Extract ligand representations ──
    print("\n" + "=" * 80)
    print("Extracting ligand representations")
    print("=" * 80)

    morgan_dict = compute_ligand_fingerprints(df)
    chemberta_dict = extract_learned_ligand_embeddings(df)

    ligand_reps = OrderedDict([
        ("Morgan",    {"dict": morgan_dict,    "dim": 2048}),
        ("ChemBERTa", {"dict": chemberta_dict, "dim": 384}),
    ])

    # ── Extract protein representations ──
    print("\n" + "=" * 80)
    print("Extracting protein representations")
    print("=" * 80)

    # Per-residue (last layer) for mean/max/attention pooling
    residue_embeds = {}
    for label in ESM_CONFIGS:
        residue_embeds[label] = extract_residue_embeddings(df, label)

    # All-layer mean-pooled for LayerW experiments
    layer_embeds = {}
    for label in ESM_CONFIGS:
        layer_embeds[label] = extract_all_layer_embeddings(df, label)

    # Pre-pool for Ridge and standard MLP
    pooled = {}
    for label in ESM_CONFIGS:
        for method in ("mean", "max"):
            pooled[(label, method)] = pool_residues(
                residue_embeds[label], method)

    # ── Run experiments ──
    all_results = []

    # ═══ Part A: Ridge (mean/max × ligand × scale) ═══════════════════════════
    print("\n" + "=" * 80)
    print("Part A: Ridge probes (mean/max pooling)")
    print("=" * 80)

    for label in ESM_CONFIGS:
        for pool in ("mean", "max"):
            prot_dict = pooled[(label, pool)]
            for lig_name, lig_info in ligand_reps.items():
                name = f"Ridge | {label} | {pool} | {lig_name}"
                print(f"\n  {name}")
                X = build_feature_matrix(df, prot_dict, lig_info["dict"])
                X_tr_full = np.concatenate([X[train_idx], X[val_idx]])

                ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
                ridge.fit(X_tr_full, y_train_val)
                y_pred = ridge.predict(X[test_idx])

                r = evaluate_predictions(y_test, y_pred, name)
                all_results.append(r)
                print(f"    CI={r['ci']:.4f} | MSE={r['mse']:.4f} | "
                      f"r={r['r']:.4f} | R²={r['r2']:.4f}")

    # ═══ Part B: MLP [512,256] (mean/max × ligand × scale) ═══════════════════
    print("\n" + "=" * 80)
    print("Part B: MLP probes (mean/max pooling)")
    print("=" * 80)

    for label in ESM_CONFIGS:
        for pool in ("mean", "max"):
            prot_dict = pooled[(label, pool)]
            for lig_name, lig_info in ligand_reps.items():
                name = f"MLP | {label} | {pool} | {lig_name}"
                print(f"\n  {name}")
                X = build_feature_matrix(df, prot_dict, lig_info["dict"])

                model = train_mlp_probe(X[train_idx], y_train,
                                        X[val_idx], y_val)
                y_pred = predict_mlp(model, X[test_idx])

                r = evaluate_predictions(y_test, y_pred, name)
                all_results.append(r)
                print(f"    CI={r['ci']:.4f} | MSE={r['mse']:.4f} | "
                      f"r={r['r']:.4f} | R²={r['r2']:.4f}")
                del model
                torch.cuda.empty_cache()

    # ═══ Part C: MLP [512,256] with attention pooling ═════════════════════════
    print("\n" + "=" * 80)
    print("Part C: MLP probes (attention pooling)")
    print("=" * 80)

    for label, cfg in ESM_CONFIGS.items():
        for lig_name, lig_info in ligand_reps.items():
            name = f"MLP | {label} | attn | {lig_name}"
            print(f"\n  {name}")
            model = train_attention_mlp(
                df, train_idx, val_idx,
                residue_embeds[label], lig_info["dict"],
                embed_dim=cfg["dim"], ligand_dim=lig_info["dim"])
            y_pred = predict_attention(
                model, df, test_idx,
                residue_embeds[label], lig_info["dict"])

            r = evaluate_predictions(y_test, y_pred, name)
            all_results.append(r)
            print(f"    CI={r['ci']:.4f} | MSE={r['mse']:.4f} | "
                  f"r={r['r']:.4f} | R²={r['r2']:.4f}")
            del model
            torch.cuda.empty_cache()

    # ═══ Part D: LayerW + MLP [512,256] ═══════════════════════════════════════
    print("\n" + "=" * 80)
    print("Part D: Layer-weighted MLP probes")
    print("=" * 80)

    for label, cfg in ESM_CONFIGS.items():
        num_layers = cfg["layers"] + 1  # includes embedding layer 0
        for lig_name, lig_info in ligand_reps.items():
            name = f"LayerW+MLP | {label} | mean | {lig_name}"
            print(f"\n  {name}")
            model = train_layer_weighted_probe(
                df, train_idx, val_idx,
                layer_embeds[label], lig_info["dict"],
                num_layers=num_layers, embed_dim=cfg["dim"],
                ligand_dim=lig_info["dim"])
            y_pred = predict_layer_weighted(
                model, df, test_idx,
                layer_embeds[label], lig_info["dict"],
                num_layers=num_layers)

            r = evaluate_predictions(y_test, y_pred, name)
            all_results.append(r)
            print(f"    CI={r['ci']:.4f} | MSE={r['mse']:.4f} | "
                  f"r={r['r']:.4f} | R²={r['r2']:.4f}")

            weights = torch.softmax(
                model.layer_weights.data.cpu(), dim=0).numpy()
            top3 = np.argsort(weights)[-3:][::-1]
            wstr = ", ".join(f"L{i}={weights[i]:.3f}" for i in top3)
            print(f"    Top-3 layers: {wstr}")
            del model
            torch.cuda.empty_cache()

    # ═══ Summary Table ════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FULL ABLATION TABLE")
    print("=" * 80)
    header = (f"{'Probe':<45s} | {'CI':>6s} | {'MSE':>6s} | "
              f"{'R²':>6s} | {'r':>6s}")
    print(header)
    print("-" * 80)
    for r in all_results:
        print(f"  {r['name']:<43s} | {r['ci']:>6.4f} | {r['mse']:>6.4f} | "
              f"{r['r2']:>6.4f} | {r['r']:>6.4f}")

    # ═══ Bottleneck Analysis ══════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    print("Effect of each component (mean CI across configurations):\n")

    def extract_factors(name):
        parts = [p.strip() for p in name.split("|")]
        return {"probe": parts[0], "scale": parts[1],
                "pool": parts[2], "ligand": parts[3]}

    for factor in ["probe", "scale", "pool", "ligand"]:
        grouped = {}
        for r in all_results:
            f = extract_factors(r["name"])
            key = f[factor]
            grouped.setdefault(key, []).append(r["ci"])

        means = {k: np.mean(v) for k, v in grouped.items()}
        delta = max(means.values()) - min(means.values())
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        print(f"  {factor.upper():<8s}: spread={delta:.4f}  "
              f"(best={best}, worst={worst})")
        for k, v in sorted(means.items(), key=lambda x: -x[1]):
            print(f"    {k:<15s}: mean CI = {v:.4f}")
        print()

    print("The component with the largest spread is the primary bottleneck.")
    print("Larger spread = more room for improvement by changing that component.")


if __name__ == "__main__":
    main()
