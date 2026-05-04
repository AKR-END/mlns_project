# Probing Frozen ESM-2 for Protein-Ligand Binding Affinity

Layer-level and residue-level analysis of frozen ESM-2 protein
representations for binding-affinity prediction, plus a final
prediction model achieving Pearson r = 0.813 on CASF-2016 with a
1.84M-parameter head atop frozen ESM-2 650M.

The full project report is in [`report.md`](./report.md).

## Research questions

1. **Where in ESM-2 is the binding signal?** — layer-wise probing
   across 8M / 35M / 650M variants × 3 ligand encoders.
2. **How does the ligand encoder shape what the model learns?** —
   Morgan ECFP4, ChemBERTa, and MolFormer compared on predictive
   accuracy and pocket localization.
3. **How do training-source and architectural choices interact with
   out-of-distribution generalization?** — `jglaser/binding_affinity`
   vs PDBbind refined-minus-core training; CASF-2016 vs CSAR-HiQ_36
   evaluation.

## Data

| Split | Source | Size | Use |
|-------|--------|-----:|-----|
| Train (jglaser variants) | `jglaser/binding_affinity` (HuggingFace) | 100k samples | Training |
| Train (PDBbind variants) | PDBbind v2016 refined-minus-core | 3,390 complexes | Training (90/10 random split for early stopping) |
| Validation | PDBbind v2016 refined-minus-core | 3,767 (jglaser runs) / 377 (PDBbind runs) | Early stopping |
| **Test 1** | PDBbind v2016 core / CASF-2016 | 290 complexes | Primary headline |
| **Test 2** | CSAR-HiQ_36 | 36 complexes | Out-of-distribution generalization & BAPULM comparison |

PDBbind v2016 must be downloaded from http://www.pdbbind.org.cn/casf.php
and unpacked at `${SCRATCH}/data/pdbbind_v2016/v2016/`. `jglaser/binding_affinity`
is loaded from HuggingFace cache via the parquet shard. The mirrored
test CSVs from BAPULM live in [`data_extra/`](./data_extra/) with
attribution.

## Repository layout

```
.
├── README.md, LICENSE, report.md, requirements.txt
│
├── src/                         # Python source
│   ├── utils.py                 # shared: data loading, ESM-2/ligand
│   │                            #   encoders, datasets, model definitions,
│   │                            #   training loop, metrics, pocket parsing
│   ├── run_analysis.py          # §4.1 layer probing + §4.2 residue interp.
│   ├── aggregate_ligands.py     # combines per-ligand results into the
│   │                            #   §4.1/§4.2 figures used in the report
│   ├── run_final_model.py       # §4.3 final model — jglaser-trained
│   ├── run_final_model_pdbbind.py # §4.3 final model — PDBbind-trained
│   ├── run_bapulm_replication.py  # §4.4 BAPULM-style head with ESM-2 650M
│   │                            #   (NOT a faithful BAPULM reproduction;
│   │                            #    encoder is substituted)
│   ├── leakage_audit.py         # §4.4 audit of BAPULM CSVs vs jglaser-100k
│   └── eval_existing_on_csars.py  # §4.4 evaluate trained checkpoints on
│                                #   the three BAPULM-released CSVs
│
├── scripts/                     # SLURM launchers
│   ├── run_analysis.sh          # one run, ligand chosen via --ligand
│   ├── run_analysis_morgan.sh   # convenience wrappers per ligand
│   ├── run_analysis_chemberta.sh
│   ├── run_analysis_molformer.sh
│   ├── run_final_model.sh       # final model (jglaser); LIGAND/SETTING env
│   ├── run_final_model_pdbbind.sh # final model (PDBbind); same env config
│   ├── run_bapulm.sh            # BAPULM-style head training
│   └── eval_existing_on_csars.sh  # evaluation grid on BAPULM-released CSVs
│
├── data_extra/                  # BAPULM-released test CSVs (mirrored
│   ├── README.md                #   from github.com/radh55sh/BAPULM)
│   ├── Test2016_290.csv
│   ├── CSAR-HiQ_36.csv
│   └── benchmark1k2101.csv
│
├── results/                     # all JSON / Markdown outputs
│   ├── layer_results_*.json     # §4.1 per-layer Ridge probe CIs
│   ├── layer_summary_*.json     # §4.1 peak layer + spread per ESM size
│   ├── interp_results_*.json    # §4.2 aggregated interpretability metrics
│   ├── interp_per_complex_*.json # §4.2 per-complex residue-level details
│   ├── final_model_*.json       # §4.3 jglaser-trained model results
│   ├── final_pdbbind_*.json     # §4.3 PDBbind-trained model results
│   ├── bapulm_replication_*.json # §4.4 BAPULM-style head outputs
│   ├── bapulm_csv_eval_results.json # §4.4 grid: 8 models × 3 BAPULM CSVs
│   ├── leakage_summary.json     # §4.4 per-test-set overlap statistics
│   └── leakage_examples.md      # §4.4 concrete leaked PDB IDs
│
└── figures/                     # all PNG figures
    ├── layer_analysis_all_ligands.png
    ├── interpretability_all_ligands.png
    ├── faithfulness_all_ligands.png
    ├── final_model_scatter_*.png, final_pdbbind_scatter_*.png
    ├── bapulm_replication_scatter.png
    ├── leakage_audit.png
    ├── sequence_attention_heatmaps_*.png  (supplementary)
    └── length_analysis.png                (supplementary)
```

All Python scripts auto-detect the project root from their own
location (so they work whether you call them as `src/run_analysis.py`
or copy them elsewhere — see `PROJECT_ROOT` in `src/utils.py`).
SLURM scripts in `scripts/` use `cd "$(dirname "$0")/.."` so they
resolve relative to the project root. Submit them from anywhere with
`sbatch scripts/<name>.sh`.

## Setup

```bash
# Tested with Python 3.12, CUDA 12.x, NVIDIA GPU with ≥10 GB VRAM
python -m venv mlns-env
source mlns-env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `torch`, `fair-esm`, `transformers`,
`rdkit-pypi`, `biopython`, `scikit-learn`, `pyarrow`, `matplotlib`,
`pandas`, `scipy`.

You also need:
- **PDBbind v2016** downloaded and unpacked at `${SCRATCH}/data/pdbbind_v2016/v2016/`
  with the standard `<pdbid>/<pdbid>_protein.pdb`, `_ligand.sdf`, and
  `_pocket.pdb` layout, plus the `index/INDEX_*` affinity files.
- **`jglaser/binding_affinity`** parquet pulled into the HuggingFace cache
  (`${HF_HOME}/datasets--jglaser--binding_affinity/...`); the loader in
  `utils.py` reads the parquet directly to bypass HF's removed
  script-loader path.
- ESM-2 weights — auto-downloaded by `fair-esm` to `${TORCH_HOME}` /
  `${CACHE_DIR}` on first use; subsequent runs reuse the cache.

`utils.py` controls all paths via `SCRATCH = "/ssd_scratch/akr"`. Set
`SCRATCH` to your own scratch directory before running.

## Reproducing the report

The end-to-end pipeline is:

All commands below are run from the project root.

### 1. Layer probing (§4.1) and residue-level interpretability (§4.2)

Per ligand encoder (Morgan, ChemBERTa, MolFormer):
```bash
python src/run_analysis.py --ligand Morgan
python src/run_analysis.py --ligand ChemBERTa
python src/run_analysis.py --ligand MolFormer
```
or via SLURM (`sbatch scripts/run_analysis_morgan.sh`, etc.).

Then aggregate (writes outputs to `figures/`):
```bash
python src/aggregate_ligands.py
```
Produces `figures/layer_analysis_all_ligands.png`,
`figures/interpretability_all_ligands.png`,
`figures/faithfulness_all_ligands.png`.

### 2. Final affinity model (§4.3)

Eight (training-source × ligand × Setting) configurations:

```bash
# jglaser-trained (the diversity-rich training set)
LIGAND=Morgan    SETTING=B sbatch scripts/run_final_model.sh
LIGAND=Morgan    SETTING=C sbatch scripts/run_final_model.sh
LIGAND=MolFormer SETTING=B sbatch scripts/run_final_model.sh
LIGAND=MolFormer SETTING=C sbatch scripts/run_final_model.sh

# PDBbind-refined-trained (the standard PDBbind benchmark training pool)
LIGAND=Morgan    SETTING=B sbatch scripts/run_final_model_pdbbind.sh
LIGAND=Morgan    SETTING=C sbatch scripts/run_final_model_pdbbind.sh
LIGAND=MolFormer SETTING=B sbatch scripts/run_final_model_pdbbind.sh
LIGAND=MolFormer SETTING=C sbatch scripts/run_final_model_pdbbind.sh
```

Setting B (LayerW + MLP, 1.84M params) ≈ 2–10 min per run depending on
training source. Setting C (LayerW + ProtQuerySelfCrossAttn + MLP,
36.6M params) ≈ 15 min on PDBbind, ≈ 4 hr on jglaser-100k. Each run
saves `results/final_*_results_*.json`,
`results/final_*_predictions_*.json`, and
`figures/final_*_scatter_*.png`.

### 3. Comparison architecture (BAPULM-style head; §4.4)

```bash
sbatch scripts/run_bapulm.sh
```
Uses ESM-2 650M (not ProtT5) so this is a comparison architecture,
**not a faithful reproduction** of BAPULM. Produces
`results/bapulm_replication_*.json` and
`figures/bapulm_replication_scatter.png`.

### 4. Comparison on BAPULM-released CSVs (§4.4)

```bash
python src/leakage_audit.py                  # writes results/leakage_summary.json,
                                             #         results/leakage_examples.md,
                                             #         figures/leakage_audit.png
sbatch scripts/eval_existing_on_csars.sh     # writes results/bapulm_csv_eval_results.json
```

`leakage_audit.py` is the basis for §4.4's overlap percentages; it
loads the first 100k rows of `jglaser/binding_affinity` and the three
BAPULM CSVs from `data_extra/` and computes per-test-set
(protein, ligand) pair / sequence / SMILES overlap.

`eval_existing_on_csars.py` auto-discovers all final-model
checkpoints under `${SCRATCH}/checkpoints/final_*` and evaluates
them on every BAPULM CSV with bootstrap 95% confidence intervals.

## Compute

All experiments run on a single NVIDIA GPU with ≥10 GB VRAM. The
batch sizes in `utils.py` (`BS_RES = 32` train, `BS_PRED_RES = 8`
inference for the residue-attention path) are tuned for 10 GB. Total
compute budget for the report is approximately **15–20 GPU-hours**
including ESM-2 embedding extraction (a one-time ~3 hr cost across
all 12,994 unique jglaser proteins) and all training runs.

Memory bottleneck: the residue-level self-attention path is O(L²) in
protein length L; with L up to 1022 and 1280-d embeddings, batch
sizes are constrained on a 10 GB GPU. Setting B avoids this and is
substantially faster.

## License

MIT — see [LICENSE](./LICENSE).

## Citation

If this code or its analyses are useful in your work, please cite:

```bibtex
@misc{rajesh2026probing,
  title  = {Probing Frozen ESM-2 for Protein-Ligand Binding Affinity},
  author = {Rajesh, Ananth Keshav},
  year   = {2026},
  note   = {MLNS course project report.}
}
```

Key prior work referenced in the report:

- **BAPULM** (the closest published sequence-only baseline):
  Meda, R. S., & Farimani, A. B. (2024). *BAPULM: Binding Affinity
  Prediction using Language Models.* arXiv:2411.04150.
  https://github.com/radh55sh/BAPULM
- **ESM-2**: Lin, Z., et al. (2023). *Evolutionary-scale prediction of
  atomic-level protein structure.* Science, 379(6637).
  https://doi.org/10.1126/science.ade2574
- **MolFormer**: Ross, J., et al. (2022). *Large-scale chemical
  language representations capture molecular structure and properties.*
  Nat. Mach. Intell., 4:1256-1264.
  https://doi.org/10.1038/s42256-022-00580-7
- **ChemBERTa**: Chithrananda, S., et al. (2020). arXiv:2010.09885.
- **PDBbind**: Wang, R., et al. (2004). *J. Med. Chem.* 47:2977-2980.
- **CASF-2016**: Su, M., et al. (2019). *J. Chem. Inf. Model.* 59:895-913.

See [`report.md`](./report.md) §9 for the full reference list with DOIs.

## Acknowledgment of AI tools

[Anthropic Claude](https://www.anthropic.com/claude) was used as a
coding assistant during script development and report drafting. All
quantitative claims in the report are independently computed by the
scripts in this repository and verified against on-disk JSON outputs.
No figures in this repository were generated by AI image-generation
tools; all are matplotlib renderings of real on-disk results.
