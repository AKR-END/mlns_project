# Probing Frozen ESM-2 Embeddings for Protein-Ligand Binding Affinity

## Objective

Test whether frozen protein language model (ESM-2) embeddings contain extractable binding affinity signal, and how much of it is linearly vs nonlinearly encoded.

## Dataset

**Davis** kinase-drug binding affinity benchmark (via [DeepDTA](https://github.com/hkmztrk/DeepDTA)):
- 30,056 protein-ligand pairs (442 kinases x 68 drugs)
- Target: pKd = -log10(Kd / 1e9)
- Split: 64% train / 16% val / 20% test (random, seed=42)

## Representations

**Protein (compared):**
- ESM-2 frozen embeddings (mean-pooled over residues, last layer) — tested 3 sizes: 8M (320d), 35M (480d), 650M (1280d)
- Baseline: amino acid composition (20d) + physicochemical descriptors (8d)

**Ligand (fixed across all experiments):**
- Morgan fingerprint, 2048-bit, radius=2 (RDKit)

## Experiments

### Experiment 1: Feasibility (Linear Probing)

Ridge regression (RidgeCV) on `[protein_emb || ligand_fp]` → pKd. Includes controls.

| Probe | Pearson r | MSE | R² | CI |
|-------|-----------|-----|----|-----|
| ESM-2 (8M) + Morgan FP | 0.593 | 0.531 | 0.351 | 0.795 |
| AA Composition + Morgan FP | 0.529 | 0.590 | 0.279 | — |
| Morgan FP only (ligand-only) | — | 1.026 | -0.253 | — |
| ESM-2 only (protein-only) | 0.315 | 0.738 | 0.098 | — |
| AA Comp only (protein-only) | 0.188 | 0.790 | 0.035 | — |
| Shuffled labels (control) | -0.043 | 0.823 | -0.006 | — |

**Findings:** Signal is real (shuffle control ~0), linearly extractable (r=0.59), and ESM-2 adds meaningful value over AA composition (+0.063 r).

### Experiment 2: Model Size Scaling (Linear Probe)

| Model | Params | Embed dim | CI | MSE |
|-------|--------|-----------|-----|-----|
| ESM-2 8M | 8M | 320 | 0.7949 | 0.5313 |
| ESM-2 35M | 35M | 480 | 0.7980 | 0.5284 |
| ESM-2 650M | 650M | 1280 | 0.7975 | 0.5285 |

**Finding:** Linearly extractable signal saturates completely — 81x more parameters gives zero improvement. The bottleneck is the linear probe, not the model.

### Experiment 3: MLP Probes across Model Sizes

MLP probes (PyTorch, Adam, early stopping on val set) on last-layer embeddings + Morgan FP.

| Probe | 8M CI | 8M MSE | 35M CI | 35M MSE | 650M CI | 650M MSE |
|-------|-------|--------|--------|---------|---------|----------|
| Ridge (linear) | 0.7949 | 0.5313 | 0.7980 | 0.5284 | 0.7975 | 0.5285 |
| MLP [256] | 0.7785 | 0.5602 | 0.8227 | 0.4324 | 0.8276 | 0.4349 |
| MLP [512] | 0.7769 | 0.5611 | 0.8328 | 0.4015 | 0.8495 | 0.3448 |
| MLP [512, 256] | 0.8451 | 0.3469 | 0.8476 | 0.3492 | 0.8601 | 0.2956 |

**Findings:**
- MLP probes unlock substantially more signal than linear probes (CI: 0.795 → 0.860 for 650M with MLP [512,256])
- With nonlinear probes, larger models DO help: 8M→650M goes from CI 0.845 to 0.860
- The information is present in larger models but encoded nonlinearly

### Experiment 4: Learned Layer Weighting

Learns a softmax-weighted combination of all ESM-2 layer embeddings (ELMo-style scalar mix), with either a linear or MLP head.

| Probe | 8M CI | 8M MSE | 35M CI | 35M MSE | 650M CI | 650M MSE |
|-------|-------|--------|--------|---------|---------|----------|
| LayerW + Linear | 0.7844 | 0.5520 | 0.7856 | 0.5473 | 0.7913 | 0.5356 |
| LayerW + MLP [512, 256] | 0.8569 | 0.3165 | 0.8673 | 0.2942 | **0.8713** | **0.2676** |

**Findings:**
- Layer weighting alone with a linear head doesn't help (slightly worse than fixed last-layer Ridge)
- Layer weighting + MLP is the best overall configuration, reaching CI=0.871 with 650M
- For 8M, layers 4-6 (later layers) dominate (L5=0.194, L4=0.160, L6=0.139)
- For 35M/650M, weights are more uniformly distributed — information spread across layers

## Summary: MLP Probes (CI and MSE)

| Probe | 8M CI | 8M MSE | 35M CI | 35M MSE | 650M CI | 650M MSE |
|-------|-------|--------|--------|---------|---------|----------|
| Ridge (linear) | 0.7949 | 0.5313 | 0.7980 | 0.5284 | 0.7975 | 0.5285 |
| MLP [256] | 0.7785 | 0.5602 | 0.8227 | 0.4324 | 0.8276 | 0.4349 |
| MLP [512] | 0.7769 | 0.5611 | 0.8328 | 0.4015 | 0.8495 | 0.3448 |
| MLP [512, 256] | 0.8451 | 0.3469 | 0.8476 | 0.3492 | **0.8601** | **0.2956** |

## Summary: Learned Layer Weighting (CI and MSE)

| Probe | 8M CI | 8M MSE | 35M CI | 35M MSE | 650M CI | 650M MSE |
|-------|-------|--------|--------|---------|---------|----------|
| LayerW + Linear | 0.7844 | 0.5520 | 0.7856 | 0.5473 | 0.7913 | 0.5356 |
| LayerW + MLP [512, 256] | 0.8569 | 0.3165 | 0.8673 | 0.2942 | **0.8713** | **0.2676** |

## Key Takeaways

1. Frozen ESM-2 embeddings contain substantial binding affinity signal
2. This signal is primarily encoded **nonlinearly** — linear probes plateau at CI~0.795 regardless of model size
3. With nonlinear probes, **larger models matter** — 650M significantly outperforms 8M
4. **Combining all layers** (learned weighting) further improves over using only the last layer
5. A 2-layer MLP + layer weighting on frozen 650M embeddings reaches CI=0.871 and MSE=0.268, approaching SOTA (CI~0.90, MSE~0.19)

## Files

- `feasibility_test.py` — Data loading, ESM-2 embedding extraction, Ridge probes, baseline comparisons, controls
- `extended_probes.py` — MLP probes and learned layer weighting across all 3 model sizes

## Environment

- Python 3.12, PyTorch 2.10, fair-esm 2.0, RDKit, scikit-learn
- GPU: NVIDIA (CUDA 12.8)
- Python path: `/ssd_scratch/akr/envs/mlns/bin/python`
- Data/embeddings cached under `/ssd_scratch/akr/`
