# Probing Binding Affinity Signal in Frozen Protein Language Model Embeddings

How is protein-ligand binding affinity information encoded in frozen protein language model (ESM-2) embeddings? We systematically probe what signal exists, where it lives (layers, residues), and how much is lost at each stage of the representation pipeline.

## Research Questions

1. **What signal exists?** Can frozen ESM-2 embeddings predict binding affinity, and is the signal linear or nonlinear?
2. **Where does it live?** Which layers and which residues encode binding-relevant information?
3. **What is lost?** How much information is destroyed by mean pooling vs. preserved by attention pooling?
4. **What is the bottleneck?** At each stage (encoder, pooling, ligand rep, probe), which component limits performance?
5. **Does attention localize to binding sites?** Do learned attention weights correspond to crystallographic binding pocket residues?

## Files

### Core Experiments (run in order)

| File | What it does | Status |
|------|-------------|--------|
| `feasibility_test.py` | Linear probing (Ridge) on frozen ESM-2 8M + Morgan FP. Baselines, controls, layer-wise analysis. | Done |
| `extended_probes.py` | MLP probes and learned layer weighting (ELMo-style) on ESM-2 650M. | Done |
| `ablation_study.py` | 36-experiment ablation: 3 ESM sizes × 3 pooling × 2 ligand reps × 3 probes. Identifies bottleneck at each stage. | Done |

### Analysis

| File | What it does | Status |
|------|-------------|--------|
| `analysis.py` | Post-ablation on Davis: error by protein length (does pooling loss scale with length?), attention weight visualization. | Not run |
| `attention_models.py` | 4 attention methods × 3 settings (9 experiments). Tests protein-only, cross-attention, self-attention+pool, self+cross in MLP+Attention, MLP+LayerW, and MLP+LayerW+Attention settings. | Not run |
| `interpretability.py` | Binding pocket overlap on PDBbind: attention vs. crystallographic pocket residues, gradient attribution faithfulness, amino acid preference, enrichment statistics. | Not run |

### External Evaluation

| File | What it does | Status |
|------|-------------|--------|
| `pdbbind_eval.py` | Train on jglaser/binding_affinity (100k), test on PDBbind Test2016_290. Compare against BAPULM paper numbers (r=0.914). | Not run |
| `train_binding_affinity.py` | Full-scale training on jglaser (1.84M) with ESM-2 650M + Morgan FP MLP. | Done |

### Visualization

| File | What it does | Status |
|------|-------------|--------|
| `visualize_results.py` | 4-panel ablation figure from `results.json`: pooling effect, scale effect, ligand comparison, bottleneck ranking. | Done |

## Datasets

| Dataset | Size | Usage |
|---------|------|-------|
| Davis (DeepDTA) | 30,056 pairs (442 kinases × 68 drugs) | Primary probing dataset |
| jglaser/binding_affinity | 1.84M pairs (HuggingFace) | Large-scale training |
| PDBbind v2016 Core Set | 290 complexes with 3D structures | External benchmark + interpretability |

## Key Findings

### From Ablation Study (36 experiments)

| Factor | CI Spread | Interpretation |
|--------|-----------|----------------|
| Probe architecture | 0.077 | Primary bottleneck — nonlinear probes essential |
| Pooling strategy | 0.026 | Mean pooling loses modest signal; max pooling hurts |
| Model scale | 0.022 | Larger models help only with nonlinear probes |
| Ligand representation | 0.005 | Morgan FP vs ChemBERTa — negligible difference |

### Summary

1. Frozen ESM-2 embeddings contain substantial binding affinity signal
2. Signal is primarily **nonlinear** — linear probes plateau at CI~0.795 regardless of model size
3. With nonlinear probes, **larger models matter** — 650M significantly outperforms 8M
4. **Mean pooling** loses modest but real information (~1% CI); **max pooling** actively hurts
5. **Layer weighting** (learned combination of all layers) improves over last-layer-only
6. Best: MLP + attention pooling + ESM-2 650M + Morgan FP → CI=0.874

## Attention Architecture Comparison

`attention_models.py` tests how different attention mechanisms interact with layer weighting:

**4 attention methods:**
- **Protein-only**: `Linear(residue, 1)` — same attention regardless of ligand
- **Cross-attention**: ligand queries protein — different ligands attend to different residues
- **Self-attention + pool**: self-attention contextualizes residues before pooling
- **Self-attn + cross-attn**: full pipeline — contextualize then ligand-conditioned selection

**3 settings:**
- **A**: MLP + Attention only (last-layer per-residue → attention pool → MLP)
- **B**: MLP + Layer Weighting only (all-layer mean-pooled → layer weights → MLP)
- **C**: MLP + Layer Weighting + Attention (both branches → MLP)

## Interpretability

`interpretability.py` tests whether learned attention localizes to actual binding sites using PDBbind 3D structures:
- Parses `*_pocket.pdb` files for crystallographic binding pocket residues
- Computes enrichment: are top-attended residues disproportionately in the pocket?
- AUC-ROC: attention weights as a binary pocket classifier
- Gradient attribution: `||d(pred)/d(residue_i)||` — what actually drives predictions vs. what attention highlights
- Amino acid type analysis: which residue types are preferentially attended?

## Dependencies

- Python 3.12, PyTorch 2.10 (CUDA 12.8)
- fair-esm 2.0 (ESM-2 protein language models)
- RDKit (Morgan fingerprints, SDF/SMILES parsing)
- scikit-learn, scipy, matplotlib
- transformers (ChemBERTa), datasets (jglaser loading)
- BioPython (PDB parsing for PDBbind)

## References

- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science 2023
- **BAPULM**: Meda & Farimani, "Binding Affinity Prediction using Language Models," arXiv:2411.04150
- **Davis**: Davis et al., "Comprehensive analysis of kinase inhibitor selectivity," Nature Biotechnology 2011
- **PDBbind**: Liu et al., "PDB-wide collection of binding data," Bioinformatics 2015
