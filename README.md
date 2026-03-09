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

| Probe | Pearson r | RMSE | R² |
|-------|-----------|------|----|
| ESM-2 (8M) + Morgan FP | 0.593 | 0.729 | 0.351 |
| AA Composition + Morgan FP | 0.529 | 0.768 | 0.279 |
| Morgan FP only (ligand-only) | — | 1.013 | -0.253 |
| ESM-2 only (protein-only) | 0.315 | 0.859 | 0.098 |
| AA Comp only (protein-only) | 0.188 | 0.889 | 0.035 |
| Shuffled labels (control) | -0.043 | 0.907 | -0.006 |

**Findings:** Signal is real (shuffle control ~0), linearly extractable (r=0.59), and ESM-2 adds meaningful value over AA composition (+0.063 r).

### Experiment 2: Model Size Scaling (Linear Probe)

| Model | Params | Embed dim | Pearson r | RMSE | R² |
|-------|--------|-----------|-----------|------|----|
| ESM-2 8M | 8M | 320 | 0.5926 | 0.7289 | 0.3509 |
| ESM-2 35M | 35M | 480 | 0.5957 | 0.7269 | 0.3544 |
| ESM-2 650M | 650M | 1280 | 0.5957 | 0.7270 | 0.3543 |

**Finding:** Linearly extractable signal saturates completely — 81x more parameters gives zero improvement. The bottleneck is the linear probe, not the model.

### Experiment 3: MLP Probes across Model Sizes

MLP probes (PyTorch, Adam, early stopping on val set) on last-layer embeddings + Morgan FP.

| Probe | 8M r | 35M r | 650M r |
|-------|------|-------|--------|
| Ridge (linear) | 0.593 | 0.596 | 0.596 |
| MLP [256] | 0.560 | 0.658 | 0.707 |
| MLP [512] | 0.700 | 0.712 | 0.768 |
| MLP [512, 256] | 0.731 | 0.786 | 0.786 |

**Findings:**
- MLP probes unlock substantially more signal than linear probes (r: 0.59 → 0.79 for 35M/650M with MLP [512,256])
- With nonlinear probes, larger models DO help: 8M→650M goes from r=0.731 to r=0.786
- The information is present in larger models but encoded nonlinearly

### Experiment 4: Learned Layer Weighting

Learns a softmax-weighted combination of all ESM-2 layer embeddings (ELMo-style scalar mix), with either a linear or MLP head.

| Probe | 8M r | 35M r | 650M r |
|-------|------|-------|--------|
| LayerW + Linear | 0.572 | 0.580 | 0.591 |
| LayerW + MLP [512, 256] | 0.770 | 0.791 | **0.819** |

**Findings:**
- Layer weighting alone with a linear head doesn't help (slightly worse than fixed last-layer Ridge)
- Layer weighting + MLP is the best overall configuration, reaching r=0.819 with 650M
- For 650M, top layers are nearly uniformly weighted — suggests information is distributed across all 34 layers
- For 8M, layers 4-6 (later layers) dominate

## Summary

| Configuration | Best Pearson r | Notes |
|---------------|---------------|-------|
| Ridge (linear, last layer) | 0.596 | Saturates at 8M, no model scaling |
| MLP [512, 256] (last layer) | 0.786 | Model scaling helps (8M→650M) |
| LayerW + MLP [512, 256] | **0.819** | Best result, uses all layers |
| SOTA (deep task-specific models) | ~0.86 | Full end-to-end training |

The gap from our best probe (r=0.819) to SOTA (r~0.86) is ~0.04 — remarkably close for frozen embeddings with a lightweight probe head.

## Key Takeaways

1. Frozen ESM-2 embeddings contain substantial binding affinity signal
2. This signal is primarily encoded **nonlinearly** — linear probes plateau at r~0.59 regardless of model size
3. With nonlinear probes, **larger models matter** — 650M significantly outperforms 8M
4. **Combining all layers** (learned weighting) further improves over using only the last layer
5. A 2-layer MLP + layer weighting on frozen 650M embeddings reaches r=0.819, closing ~85% of the gap to SOTA

## Files

- `feasibility_test.py` — Data loading, ESM-2 embedding extraction, Ridge probes, baseline comparisons, controls
- `extended_probes.py` — MLP probes and learned layer weighting across all 3 model sizes

## Environment

- Python 3.12, PyTorch 2.10, fair-esm 2.0, RDKit, scikit-learn
- GPU: NVIDIA (CUDA 12.8)
- Python path: `/ssd_scratch/akr/envs/mlns/bin/python`
- Data/embeddings cached under `/ssd_scratch/akr/`
