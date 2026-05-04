"""
Evaluate our jglaser-trained models on BAPULM's released test CSVs using
BAPULM's *exact* sequence and SMILES strings (not our PDB-parsed ones).

Why: this isolates whether BAPULM's high reported numbers come from their
string-format choice. If our model — which doesn't memorize jglaser strings
particularly well — still shows a big leakage boost on benchmark1k2101
(100% in training), the format choice is the dominant effect.

Inputs (cloned from github.com/radh55sh/BAPULM into <repo>/data_extra/):
  Test2016_290.csv    — 290 complexes, 12.8% pair leak, 71% ligand leak
  CSAR-HiQ_36.csv     —  36 complexes, 0% leak (clean)
  benchmark1k2101.csv — 1000 complexes, 100% leak

Models evaluated (loaded from /ssd_scratch/akr/checkpoints/):
  Setting B (Morgan + LayerW + MLP)
  Setting C (Morgan + LayerW + ProtQuerySelfCrossAttn)
  BAPULM-style head (ESM-2 mean + MolFormer + BAPULM 4-layer MLP)

For Test2016_290 we also report clean vs leaked subsets so the per-set
leakage premium is visible per-model.

Output:
  bapulm_csv_eval_results.json
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RESULTS_DIR, FIGURES_DIR, DATA_EXTRA_DIR,
    DEVICE, ESM_CONFIGS, CKPT_DIR,
    load_jglaser, extract_esm2_embeddings,
    compute_morgan, compute_molformer,
    LayerDS, LayerResidueDS, layer_residue_collate_fn,
    LayerWMLP, LayerWAttnMLP, ATTN_METHODS, concordance_index,
)


CSVS = {
    "Test2016_290":     os.path.join(DATA_EXTRA_DIR, "Test2016_290.csv"),
    "CSAR-HiQ_36":      os.path.join(DATA_EXTRA_DIR, "CSAR-HiQ_36.csv"),
    "benchmark1k2101":  os.path.join(DATA_EXTRA_DIR, "benchmark1k2101.csv"),
}


def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=0):
    if len(y_true) < 5:
        return None
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rs, rmses, r2s, cis = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9: continue
        rs.append(stats.pearsonr(yt, yp)[0])
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        r2s.append(r2_score(yt, yp))
        cis.append(concordance_index(yt, yp))
    def ci(arr): return (float(np.percentile(arr, 2.5)),
                          float(np.percentile(arr, 97.5)))
    return {"r":    {"point": float(stats.pearsonr(y_true, y_pred)[0]), "ci95": ci(rs), "n": int(n)},
            "rmse": {"point": float(np.sqrt(mean_squared_error(y_true, y_pred))), "ci95": ci(rmses)},
            "r2":   {"point": float(r2_score(y_true, y_pred)),  "ci95": ci(r2s)},
            "ci":   {"point": float(concordance_index(y_true, y_pred)), "ci95": ci(cis)}}


def fmt(m):
    if m is None: return "n/a (n<5)"
    lo, hi = m["r"]["ci95"]
    return (f"r={m['r']['point']:+.3f} [{lo:+.3f},{hi:+.3f}]  "
            f"RMSE={m['rmse']['point']:.3f}  "
            f"CI={m['ci']['point']:.3f}  "
            f"R2={m['r2']['point']:+.3f}  n={m['r']['n']}")


def load_jglaser_pairs(n=100000):
    parquet = sorted(glob.glob(
        "/ssd_scratch/akr/hf_cache/datasets--jglaser--binding_affinity/"
        "snapshots/*/data/*.parquet"))[0]
    pf = pq.ParquetFile(parquet)
    chunks, taken = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["seq", "smiles_can"]).to_pandas()
        if taken + len(t) >= n:
            chunks.append(t.iloc[: n - taken]); break
        chunks.append(t); taken += len(t)
    df = pd.concat(chunks, ignore_index=True)
    return set(zip(df.seq.astype(str), df.smiles_can.astype(str)))


def main():
    # Normalization stats: same as the jglaser-100k training run.
    print("Loading jglaser-100k normalization stats...")
    _, _, train_targets, _, _ = load_jglaser(100000)
    t_mean, t_std = float(train_targets.mean()), float(train_targets.std())
    print(f"  t_mean={t_mean:.4f}  t_std={t_std:.4f}")

    # For Test2016_290 leakage analysis (per-model)
    print("\nIndexing jglaser-100k pairs for leakage filter...")
    jg_pairs = load_jglaser_pairs(100000)
    print(f"  unique training (seq, smiles) pairs: {len(jg_pairs)}")

    cfg = ESM_CONFIGS["650M"]
    embed_dim = cfg["dim"]
    num_layers = cfg["layers"] + 1

    all_results = {}

    for csv_name, csv_path in CSVS.items():
        print("\n" + "=" * 80)
        print(f"# {csv_name}  ({csv_path})")
        print("=" * 80)

        df = pd.read_csv(csv_path)
        # benchmark1k2101.csv has no pdbid column; synthesize codes from row index.
        if "pdbid" in df.columns:
            codes = [str(x) for x in df["pdbid"].tolist()]
        else:
            codes = [f"{csv_name}_{i:05d}" for i in range(len(df))]
        seqs  = df["seq"].astype(str).tolist()
        smis  = df["smiles_can"].astype(str).tolist()
        aff   = df["neg_log10_affinity_M"].to_numpy(dtype=np.float32)
        n = len(codes)
        print(f"  n={n}")

        # Per-row leakage flag against jglaser-100k pairs
        is_leaked = np.array([(s, m) in jg_pairs for s, m in zip(seqs, smis)])
        print(f"  pair leakage: {is_leaked.sum()}/{n} "
              f"({100*is_leaked.sum()/n:.1f}%)")

        # ── Embeddings (cache per CSV) ──
        cache_tag = f"bapulm_{csv_name.replace('-', '_').lower()}"
        print(f"  cache_tag = {cache_tag}")

        # All-layer (Setting B + C) and per-residue (Setting C, BAPULM head also
        # uses last-layer mean-pool, which we get from the all-layer dict).
        layer_dict = extract_esm2_embeddings(seqs, codes, cache_tag, "650M",
                                             all_layers=True)
        res_dir    = extract_esm2_embeddings(seqs, codes, cache_tag, "650M",
                                             per_residue=True)
        morgan_dict   = compute_morgan(smis, codes)
        molformer_dict = compute_molformer(smis, codes, cache_tag)

        # Mean-pooled last-layer dict for BAPULM-style head
        mean_dict = {nm: layer_dict[nm][num_layers - 1] for nm in codes}

        targets_std = (aff - t_mean) / t_std
        ligand_dim_morgan = int(len(next(iter(morgan_dict.values()))))
        ligand_dim_molformer = int(len(next(iter(molformer_dict.values()))))

        results_per_set = {
            "n": int(n),
            "n_leaked": int(is_leaked.sum()),
            "models": {},
        }

        # Generic helper to evaluate a Morgan-or-MolFormer Setting-B/C model
        def _eval_layer_model(label, ckpt, lig_dict, lig_dim, setting):
            if not os.path.exists(ckpt):
                return
            print(f"\n  {label} — {ckpt}")
            if setting == "B":
                model = LayerWMLP(num_layers, embed_dim, lig_dim).to(DEVICE)
            else:
                attn_cls = ATTN_METHODS["protq+cross"]
                model = LayerWAttnMLP(attn_cls, num_layers, embed_dim, lig_dim).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()
            if setting == "B":
                ds = LayerDS(codes, codes, targets_std, layer_dict, lig_dict, num_layers)
                preds = []
                with torch.no_grad():
                    for ly, lg, _ in DataLoader(ds, 256):
                        preds.append(model(ly.to(DEVICE), lg.to(DEVICE)).cpu())
            else:
                ds = LayerResidueDS(codes, codes, targets_std,
                                    layer_dict, res_dir, lig_dict, num_layers)
                preds = []
                with torch.no_grad():
                    for ly, rs, lg, mk, _ in DataLoader(ds, 8, collate_fn=layer_residue_collate_fn):
                        preds.append(model(ly.to(DEVICE), rs.to(DEVICE),
                                           lg.to(DEVICE), mk.to(DEVICE)).cpu())
            yp = torch.cat(preds).numpy() * t_std + t_mean
            del model; torch.cuda.empty_cache()
            results_per_set["models"][label] = {
                "all":         bootstrap_metrics(aff, yp),
                "leaked_only": bootstrap_metrics(aff[is_leaked],  yp[is_leaked]),
                "clean_only":  bootstrap_metrics(aff[~is_leaked], yp[~is_leaked]),
            }
            print(f"    all     : {fmt(results_per_set['models'][label]['all'])}")
            print(f"    leaked  : {fmt(results_per_set['models'][label]['leaked_only'])}")
            print(f"    clean   : {fmt(results_per_set['models'][label]['clean_only'])}")

        # ── Auto-discover and evaluate every trained Setting-B/C checkpoint ──
        # (jglaser-trained: final_<LIGAND>_S<X>_<attn>_650M.pt
        #  PDBbind-refined-trained: final_pdbbind_<LIGAND>_S<X>_<attn>_650M.pt)
        candidates = [
            # (label, ckpt_filename, lig_dict, lig_dim, setting)
            ("Setting B Morgan (jglaser)",         "final_Morgan_SB_na_650M.pt",                morgan_dict,    ligand_dim_morgan,    "B"),
            ("Setting C Morgan (jglaser)",         "final_Morgan_SC_protq+cross_650M.pt",       morgan_dict,    ligand_dim_morgan,    "C"),
            ("Setting B MolFormer (jglaser)",      "final_MolFormer_SB_na_650M.pt",             molformer_dict, ligand_dim_molformer, "B"),
            ("Setting C MolFormer (jglaser)",      "final_MolFormer_SC_protq+cross_650M.pt",    molformer_dict, ligand_dim_molformer, "C"),
            ("Setting B Morgan (PDBbind-refined)", "final_pdbbind_Morgan_SB_na_650M.pt",        morgan_dict,    ligand_dim_morgan,    "B"),
            ("Setting C Morgan (PDBbind-refined)", "final_pdbbind_Morgan_SC_protq+cross_650M.pt", morgan_dict,  ligand_dim_morgan,    "C"),
            ("Setting B MolFormer (PDBbind-refined)", "final_pdbbind_MolFormer_SB_na_650M.pt",  molformer_dict, ligand_dim_molformer, "B"),
            ("Setting C MolFormer (PDBbind-refined)", "final_pdbbind_MolFormer_SC_protq+cross_650M.pt", molformer_dict, ligand_dim_molformer, "C"),
        ]
        for label, ckpt_name, lig_dict, lig_dim, setting in candidates:
            _eval_layer_model(label, os.path.join(CKPT_DIR, ckpt_name),
                              lig_dict, lig_dim, setting)

        # ── BAPULM-style head (ESM-2 mean + MolFormer + BAPULM 4-layer MLP) ──
        ckpt_bap = os.path.join(CKPT_DIR, "bapulm_replication.pt")
        if os.path.exists(ckpt_bap):
            print(f"\n  BAPULM-style head (ESM-2 mean + MolFormer) — {ckpt_bap}")
            # Inline class to avoid an extra import
            import torch.nn as nn
            class BAPULMHead(nn.Module):
                def __init__(self, prot_dim=1280, mol_dim=768):
                    super().__init__()
                    self.prot_linear = nn.Linear(prot_dim, 512)
                    self.mol_linear = nn.Linear(mol_dim, 512)
                    self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
                    self.dropout = nn.Dropout(p=0.1)
                    self.linear1 = nn.Linear(1024, 768)
                    self.linear2 = nn.Linear(768, 512)
                    self.linear3 = nn.Linear(512, 32)
                    self.final_linear = nn.Linear(32, 1)
                def forward(self, prot, mol):
                    po = torch.relu(self.prot_linear(prot))
                    mo = torch.relu(self.mol_linear(mol))
                    x = torch.cat([po, mo], dim=1)
                    x = self.norm(x); x = self.dropout(x)
                    x = torch.relu(self.linear1(x))
                    x = torch.relu(self.linear2(x)); x = self.dropout(x)
                    x = torch.relu(self.linear3(x))
                    return self.final_linear(x).squeeze(-1)

            model = BAPULMHead(prot_dim=embed_dim, mol_dim=ligand_dim_molformer).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_bap, map_location="cpu",
                                             weights_only=True))
            model.eval()

            preds = []
            with torch.no_grad():
                for i in range(0, n, 256):
                    batch_codes = codes[i:i+256]
                    p = torch.stack([mean_dict[c] if isinstance(mean_dict[c], torch.Tensor)
                                      else torch.from_numpy(mean_dict[c])
                                      for c in batch_codes]).float().to(DEVICE)
                    m = torch.stack([torch.from_numpy(molformer_dict[c]) if isinstance(molformer_dict[c], np.ndarray)
                                      else molformer_dict[c]
                                      for c in batch_codes]).float().to(DEVICE)
                    preds.append(model(p, m).cpu())
            yp = torch.cat(preds).numpy() * t_std + t_mean
            del model; torch.cuda.empty_cache()
            results_per_set["models"]["BAPULM head (MolFormer)"] = {
                "all":         bootstrap_metrics(aff, yp),
                "leaked_only": bootstrap_metrics(aff[is_leaked],  yp[is_leaked]),
                "clean_only":  bootstrap_metrics(aff[~is_leaked], yp[~is_leaked]),
            }
            print(f"    all     : {fmt(results_per_set['models']['BAPULM head (MolFormer)']['all'])}")
            print(f"    leaked  : {fmt(results_per_set['models']['BAPULM head (MolFormer)']['leaked_only'])}")
            print(f"    clean   : {fmt(results_per_set['models']['BAPULM head (MolFormer)']['clean_only'])}")

        all_results[csv_name] = results_per_set

    out = os.path.join(RESULTS_DIR, "bapulm_csv_eval_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved {out}")

    # ── Final summary table ──
    print("\n" + "=" * 90)
    print("SUMMARY (Pearson r on BAPULM-released CSVs, our jglaser-trained models)")
    print("=" * 90)
    print(f"  {'CSV':<22s} {'leak%':>6s} {'Setting B':>14s} {'Setting C':>14s} {'BAPULM head':>14s}")
    for name, info in all_results.items():
        leak_pct = 100 * info["n_leaked"] / info["n"]
        rB = info["models"].get("Setting B (Morgan)", {}).get("all", {}).get("r", {}).get("point")
        rC = info["models"].get("Setting C (Morgan)", {}).get("all", {}).get("r", {}).get("point")
        rA = info["models"].get("BAPULM head (MolFormer)", {}).get("all", {}).get("r", {}).get("point")
        print(f"  {name:<22s} {leak_pct:>5.1f}% "
              f"{(f'{rB:+.3f}' if rB is not None else '-'):>14s} "
              f"{(f'{rC:+.3f}' if rC is not None else '-'):>14s} "
              f"{(f'{rA:+.3f}' if rA is not None else '-'):>14s}")


if __name__ == "__main__":
    main()
