# Probing Binding Affinity Signal in Frozen Protein Language Model Embeddings

How is protein-ligand binding affinity information encoded in frozen protein language model (ESM-2) embeddings? We systematically probe what signal exists, where it lives (layers, residues), and how much is lost at each stage of the representation pipeline.

## Research Questions

1. **What signal exists?** Can frozen ESM-2 embeddings predict binding affinity, and is the signal linear or nonlinear?
2. **Where in the network?** Which ESM-2 layers encode binding-relevant information, and how does this change with model scale?
3. **Where in the sequence?** Do learned attention weights localize to crystallographic binding pocket residues?
4. **What is lost?** How much information is destroyed by mean pooling, and does this scale with protein length?
5. **What is the bottleneck?** At each stage (encoder, pooling, ligand rep, probe), which component limits performance?

## Data Splits

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Train | jglaser/binding_affinity (HuggingFace) | 100k samples | Training all models |
| Val | PDBbind v2016 refined set minus core | 3,767 complexes | Early stopping / model selection |
| Test | PDBbind v2016 core set (CASF-2016) | 290 complexes | Final evaluation |

## Files

### Core

| File | What it does |
|------|-------------|
| `utils.py` | All shared code: data loading, ESM-2 embedding extraction, Morgan FP computation, model definitions (MLPProbe, 4 attention modules, 3 settings), datasets, training loops, metrics, pocket parsing |
| `feasibility_test.py` | Original feasibility check on Davis (linear probing, baselines, controls). Kept as reference for early results |

### Experiments

| File | What it probes | Outputs |
|------|---------------|---------|
| `run_ablation.py` | **Step 1: Lock down results.** 36-experiment ablation: 3 ESM sizes × 3 pooling (mean/max/attn) × Ridge/MLP + LayerW+MLP. Computes CI spread per factor, identifies bottleneck | `ablation_results.json`, `ablation_plots.png` |
| `run_analysis.py` | **Steps 2–4 + attention architectures.** Layer analysis, residue-level interpretability, length analysis, 4×3 attention comparison | `layer_analysis.png`, `interpretability.png`, `length_analysis.png`, `attention_comparison.png` |

### Legacy (in `archive/`)

Previous iteration scripts superseded by the consolidated versions above. Kept for reference.

## Step 1: Ablation Study (`run_ablation.py`)

36 experiments crossing:
- **Protein encoder**: ESM-2 {8M, 35M, 650M}
- **Pooling**: {mean, max, learned attention}
- **Probe**: {Ridge (linear), MLP [512,256], LayerW+MLP [512,256]}
- **Ligand**: Morgan FP (2048-bit)

**Deliverables:**
- Full results table (CI, Pearson r, RMSE, MSE)
- CI spread per factor → bottleneck ranking
- 3 plots: pooling comparison, scale vs performance, linear vs nonlinear gap
- Takeaways: e.g. "Nonlinear decoding recovers +X CI → signal is not linearly accessible"

## Step 2: Layer-Level Analysis (in `run_analysis.py`)

Per-layer Ridge probes for all 3 ESM-2 sizes. Answers "where in the network?"

**Deliverables:**
- CI vs layer curves for 8M, 35M, 650M
- Peak layer and spread (variance) per model size
- Key result: "Small models concentrate signal in late layers, large models distribute it"

## Step 3: Residue-Level Interpretability (in `run_analysis.py`)

Compares learned attention weights against crystallographic binding pocket residues from PDBbind `*_pocket.pdb` files.

**Metrics:**
- AUC-ROC (attention weights as pocket classifier)
- Precision@K (top-K attended residues vs pocket residues)
- Pocket enrichment (observed/expected overlap)
- Gradient attribution: `||d(pred)/d(residue_i)||` — what actually drives predictions
- Attention faithfulness: Spearman correlation between attention and gradient importance

**Deliverable:** One strong figure + one clear conclusion:
- Attention localizes → **local encoding** (PLM learns binding site features from sequence)
- Attention is diffuse → **global encoding** (PLM uses whole-protein properties)

## Step 4: Pooling + Length Analysis (in `run_analysis.py`)

Performance vs protein sequence length for mean-pooled vs attention-pooled MLP.

**Deliverable:** "Pooling destroys more information for longer sequences" (or not)

## Attention Architecture Comparison (in `run_analysis.py`)

4 attention methods × 3 settings = 9 experiments:

**4 Attention Methods:**
| Method | Formula | Ligand-dependent? |
|--------|---------|-------------------|
| Protein-only | `Linear(residue, 1)` | No |
| Cross-attention | `softmax(lig_query @ prot_keys^T)` | Yes |
| Self-attn + pool | `SelfAttn(residues) → Linear(ctx, 1)` | No |
| Self-attn + cross | `SelfAttn(residues) → softmax(lig @ ctx_keys^T)` | Yes |

**3 Settings:**
| Setting | Architecture |
|---------|-------------|
| A: MLP + Attention | `[attn_pool(residues) \|\| ligand] → MLP` |
| B: MLP + Layer Weighting | `[layer_weighted(layers) \|\| ligand] → MLP` |
| C: MLP + LayerW + Attention | `[layer_weighted \|\| attn_pool \|\| ligand] → MLP` |

## BAPULM Comparison

BAPULM (Meda & Farimani, arXiv:2411.04150) reports r=0.914, RMSE=0.898 on Test2016_290 using ProtT5 + MolFormer. Our models are evaluated on the same test set for direct comparison. We use their published numbers — no BAPULM re-evaluation.

## Dependencies

- Python 3.12, PyTorch 2.10 (CUDA 12.8)
- fair-esm 2.0 (ESM-2 protein language models)
- RDKit (Morgan fingerprints, SDF/SMILES parsing)
- scikit-learn, scipy, matplotlib
- datasets (HuggingFace, for jglaser loading)
- BioPython (PDB parsing)

## References

- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science 2023
- **BAPULM**: Meda & Farimani, "Binding Affinity Prediction using Language Models," arXiv:2411.04150
- **PDBbind**: Liu et al., "PDB-wide collection of binding data," Bioinformatics 2015
- **jglaser/binding_affinity**: Glaser et al., HuggingFace dataset of protein-ligand binding affinities
