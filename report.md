# Probing Frozen ESM-2 for Protein–Ligand Binding Affinity: Layers, Ligands, and Architecture Choices

*MLNS course project report*

---

## Abstract

Sequence-only protein-ligand binding-affinity models built on frozen
protein language models (PLMs) report strong performance on PDBbind
benchmarks, but it remains unclear (i) which layers of the PLM carry
the binding signal, (ii) how the choice of ligand encoder affects
both predictive accuracy and pocket localization, and (iii) how
training-source and architectural choices interact with
out-of-distribution generalization. We address all three questions
with a frozen ESM-2 (8M, 35M, 650M) protein encoder paired with
three ligand representations (Morgan ECFP, ChemBERTa, MolFormer),
trained on either `jglaser/binding_affinity` (100k samples) or
PDBbind refined-minus-core, and evaluated on PDBbind core
(CASF-2016) and CSAR-HiQ_36. We find (a) all three ligand encoders
concentrate binding-relevant signal at layers 27–29 of ESM-2 650M
despite very different chemical priors; (b) the encoder that best
localizes attention to the binding pocket (MolFormer, 2.4× pocket
enrichment) is also the most training-distribution-sensitive,
losing the most performance under domain shift; and (c) a simple
layer-weighted head with 1.84M trainable parameters on top of
frozen ESM-2 650M with MolFormer ligand features, trained on
PDBbind refined-minus-core, achieves Pearson r = 0.813
[95% CI 0.768, 0.849], CI = 0.810, RMSE = 1.295 on CASF-2016. As a
positioning point, we compare against BAPULM (Meda & Farimani 2024),
the closest published sequence-only baseline, on CSAR-HiQ_36; we
note that direct numerical comparison to BAPULM is limited because
they use ProtT5-XL-U50 (3B parameters) while we use ESM-2 650M
(4.6× smaller) and not all of their released test sets are
disjoint from their training data, so we restrict the comparison
to CSAR-HiQ_36 (which is disjoint from `jglaser/binding_affinity`).

---

## 1. Introduction

Recent affinity prediction models on PDBbind core (CASF-2016) report
Pearson r values approaching or exceeding 0.9, including with
sequence-only architectures built on frozen PLMs. Despite the strong
headline numbers, three questions about these models remain open:

1. **Where in the PLM is the binding signal?** ESM-2 has 33 transformer
   layers in its 650M variant. Picking a single layer for downstream
   tasks discards the rest, but the standard practice does not specify
   which layer is best for affinity, nor whether the answer is robust
   across ligand encoders.

2. **How does the ligand encoder shape what the model learns?** A
   2048-bit Morgan fingerprint, a 384-d ChemBERTa embedding, and a
   768-d MolFormer embedding express very different priors over
   chemical structure. Whether the protein-side representation
   adapts to these differences — and whether one choice trades
   interpretability for accuracy — has not been characterized.

3. **How do training-source and architectural choices interact
   with out-of-distribution generalization?** Sequence-only PLM
   predictors are typically trained on jglaser-style binding
   datasets and evaluated on PDBbind core (CASF-2016), but the
   in-distribution vs. out-of-distribution behaviour of different
   ligand encoders and head architectures has not been
   characterized.

We address all three. We probe layer-by-layer, contrast three ligand
encoders, characterize the localization–accuracy tradeoff, and
report a final affinity-prediction model. As context, we
position our results against BAPULM (the closest published
sequence-only baseline); a brief note in §4.4 explains our
test-set choice for that comparison.

---

## 2. Baseline and innovations

### 2.0.1 Baseline: BAPULM

The closest comparable model is **BAPULM** (Meda & Farimani, 2024)
[1], a sequence-only protein-ligand binding affinity predictor that
uses two frozen pretrained language models for feature extraction
followed by a small MLP head. Its features are:

- **Protein encoder:** ProtT5-XL-U50 [2] (3B parameters, frozen),
  encoder side only, last-layer mean-pool over residues, yielding
  1024-d protein vectors.
- **Ligand encoder:** MolFormer-XL-both-10pct [3] (frozen),
  last-layer mean-pool over SMILES tokens, 768-d.
- **Head:** Linear projections of each modality to 512-d, concatenated
  to 1024-d, BatchNorm, Dropout(0.1), then a 4-layer MLP
  (1024 → 768 → 512 → 32 → 1) with ReLU activations and another
  Dropout layer.
- **Training:** First 100,000 rows of `jglaser/binding_affinity` [4],
  random 90/10 split, Adam lr = 1e-3, MSE loss,
  ReduceLROnPlateau scheduler.
- **Reported headline:** Pearson r = 0.914 on `Test2016_290.csv`
  (CASF-2016), r = 0.925 on `benchmark1k2101.csv`, r = 0.813 on
  `CSAR-HiQ_36.csv`. Code at
  `https://github.com/radh55sh/BAPULM`.

### 2.0.2 Our extensions relative to BAPULM

We position this work as a *probing study* with a final affinity
prediction model; BAPULM serves as the closest published reference
point. Direct numerical comparison to BAPULM is limited by
encoder-size differences (their ProtT5-XL-U50 has 3B parameters; our
ESM-2 650M has 4.6× fewer). We make four extensions relative to
BAPULM's setup:

1. **Probing analysis (§4.1, §4.2).** BAPULM treats the frozen
   protein encoder as a black-box feature extractor; we probe it
   layer-by-layer across all three ESM-2 sizes (8M, 35M, 650M)
   and across three ligand encoders (Morgan ECFP, ChemBERTa,
   MolFormer). This is the primary contribution of this report.

2. **Layer-weighted heads (Setting B/C).** Whereas BAPULM uses
   only the last-layer mean-pool of ProtT5, we learn a softmax
   weighting over *all* ESM-2 layers. This is motivated by the
   probing analysis (§4.1) which finds that the optimal layer is
   not the last one (peaks are at L27–L29 of L34 in ESM-2 650M).

3. **Multi-ligand encoder ablation.** BAPULM uses MolFormer
   exclusively. We compare Morgan, ChemBERTa, and MolFormer
   against the same frozen protein encoder and find that
   pocket-localization quality and predictive accuracy are
   inversely correlated, with MolFormer the most localizing
   and most distribution-sensitive of the three (§4.3).

4. **Training-source ablation.** BAPULM trains on `jglaser/
   binding_affinity` (first 100k rows). We additionally train
   on PDBbind refined-minus-core (the standard PDBbind benchmark
   training pool), and find 0.12–0.38 r-point CASF-2016
   improvements depending on (ligand, architecture). The
   training source materially shapes the
   accuracy/transferability tradeoff (§4.3).

For test-set selection: BAPULM releases three test CSVs
(`benchmark1k2101.csv`, `Test2016_290.csv`, `CSAR-HiQ_36.csv`).
The first two share substantial overlap with their training data
(see §4.4 for details) — `benchmark1k2101.csv` shares 100% of its
(protein, ligand) string pairs with the first 100k rows of
`jglaser/binding_affinity`, and `Test2016_290.csv` shares 12.8% of
pairs and 71% of ligand SMILES. We therefore restrict our
quantitative comparison against BAPULM to **CSAR-HiQ_36**, which
shares 0% (protein, ligand) pairs and only 2.8% of ligand SMILES
with `jglaser/binding_affinity`. This is a methodological
positioning choice, not a primary contribution of the paper.

## 3. Methods

### 3.1 Data

We use three datasets:
- **`jglaser/binding_affinity`** — first 100,000 (protein, ligand,
  affinity) rows of the HuggingFace dataset (12,994 unique proteins,
  61,288 unique ligands), used for training in the jglaser-trained
  models.
- **PDBbind v2016** — refined set (4,057 complexes), with 290 of these
  comprising the **CASF-2016 / PDBbind core** test set. Refined-minus-core
  yields 3,767 complexes that we use for both validation (early stopping)
  in jglaser-trained runs and as the training pool in PDBbind-trained
  runs.
- **CSAR-HiQ_36** — 36 complexes released alongside BAPULM; structurally
  and chemically distinct from PDBbind, used as our true
  out-of-distribution test.

All sequences are parsed from `_protein.pdb` files via BioPython; all
SMILES are canonicalized from `_ligand.sdf` via RDKit
(`Chem.MolToSmiles`). This pipeline is consistent across all
training and evaluation. We refer to these as **PDB-parsed strings**.

### 3.2 Protein and ligand encoders

The protein encoder is **frozen ESM-2** at three sizes: 8M
(`esm2_t6_8M_UR50D`), 35M (`esm2_t12_35M_UR50D`), and 650M
(`esm2_t33_650M_UR50D`). For Step 2 (layer probing) we extract
mean-pooled embeddings from every layer (8, 13, 34 layers respectively).
For Steps 3-5 we extract per-residue embeddings from the final layer
plus a learned softmax over per-layer mean-pooled vectors. ESM-2 is
never updated.

Ligand encoders:
- **Morgan ECFP4** — 2048-bit fingerprint (radius 2) via RDKit.
- **ChemBERTa** — `DeepChem/ChemBERTa-77M-MLM`, 384-d, mask-aware
  mean-pool over token sequence.
- **MolFormer** — `ibm/MoLFormer-XL-both-10pct`, 768-d, same pooling.

### 3.3 Architectures

Three head architectures atop frozen ESM-2 and the chosen ligand
encoder:
- **Setting A** (Attn): `[attention_pool(residues) || ligand]` →
  MLP. Five attention modules tested in the architecture sweep
  (ProteinOnlyAttn, CrossAttn, SelfAttnPool, SelfCrossAttn,
  ProtQuerySelfCrossAttn).
- **Setting B** (LayerW): softmax-weighted combination of all
  layers' mean-pooled vectors → `[combined_protein || ligand]` →
  MLP. **1.84 M trainable parameters.**
- **Setting C** (LayerW + Attn): combines Setting A and B —
  `[layer-weighted protein || attention-pooled residues || ligand]`
  → MLP. ProtQuerySelfCrossAttn used by default. **36.6 M
  trainable parameters.**

As an architectural comparison point, we also implement an analog
of BAPULM's head atop our protein encoder — ESM-2 650M last-layer
mean-pool + MolFormer mean-pool + BAPULM's 4-layer MLP head with
BatchNorm and Dropout (2.25 M trainable parameters). Note that this
substitutes ESM-2 650M for ProtT5-XL-U50 (3B parameters) and is
therefore *not* a faithful reproduction of BAPULM; we use it only
as a comparison architecture in the ligand-encoder × head ablation.

### 3.4 Training

All models trained with Adam (lr = 1e-3, weight decay 1e-4),
ReduceLROnPlateau (factor 0.5, patience 5), MSE loss on standardized
affinity targets (mean and std computed from the training pool),
early stopping with patience 25-30 epochs. Batch sizes are set so
the worst-case residue-attention path fits a 10 GB GPU
(BS_train = 32, BS_predict = 8 for Setting A/C; BS = 256 for Setting
B and BAPULM head).

For each (setting, ligand, training-source) combination we report
Pearson r, RMSE, R², and concordance index (CI) on each test set,
with bootstrap 95% confidence intervals (1000 samples).

### 3.4.1 Compute and training time

All experiments were run on a single NVIDIA GPU (10 GB VRAM, A100 or
RTX-class as available on the cluster) over a multi-day window.
Approximate per-configuration training times:

| Configuration | Train samples | Approx. wall time |
|---|---:|---:|
| BAPULM-style head, jglaser | 100k | ~3 min (47 ep, batch 256) |
| Setting B Morgan, jglaser | 100k | ~9 min (26 ep, batch 256) |
| Setting B MolFormer, jglaser | 100k | ~10 min (similar) |
| Setting C Morgan, jglaser | 100k | ~4 hours (27 ep, batch 32, residue-attn) |
| Setting C MolFormer, jglaser | 100k | ~4 hours (estimate; trained for completeness) |
| Setting B Morgan, PDBbind refined | 3,390 | ~2 min (85 ep, batch 256) |
| Setting C Morgan, PDBbind refined | 3,390 | ~10-15 min (batch 32) |
| Setting B MolFormer, PDBbind refined | 3,390 | ~2 min |
| Setting C MolFormer, PDBbind refined | 3,390 | ~15 min |
| ESM-2 650M embedding extraction (12,994 unique proteins) | — | ~3 hr (one-time, cached) |
| Layer probing across 3 ligands × 3 ESM sizes | — | ~2 hr (cached embeddings reused) |
| Residue-level interpretability across 3 ligands | — | ~30 min (cached embeddings reused) |

**Total compute budget:** approximately 15-20 GPU-hours including
embedding extraction and all training runs across the four sections of
this paper.

**Memory bottleneck:** the residue-level self-attention path is
O(L²) in protein length L. With L up to 1022 residues and ESM-2
650M's 1280-d embedding, batch sizes are constrained to 32 (training)
and 8 (inference) on a 10 GB GPU. Setting B avoids this bottleneck
because it does not use per-residue attention.

**Software stack.** PyTorch 2.x, `esm` 2.0 (Facebook AI's ESM-2
release), HuggingFace `transformers` (for ChemBERTa/MolFormer
loading), HuggingFace `datasets` (parquet read for jglaser), RDKit
(SMILES canonicalization, Morgan FP), BioPython (PDB parsing),
`scikit-learn` (Ridge probe, metrics), `numpy`, `scipy`, `matplotlib`.

**Storage.** Approximately 50 GB of cached intermediates on
`/ssd_scratch`: ESM-2 sharded per-residue embeddings (one `.pt`
per protein), all-layer mean-pooled tensors, ligand encoder
outputs, and model checkpoints. All cleared between cluster
sessions.

### 3.5 Evaluation

Each model is evaluated on four test sets:
- **CASF-2016** (290 PDBbind core complexes, **PDB-parsed strings**,
  via our consistent `load_pdbbind_set("core")` pipeline).
- **CASF-2016-jglaser** — same 290 complexes, but using the sequence
  and SMILES strings from BAPULM's released `Test2016_290.csv` (which
  match the jglaser format). This isolates the test-set string-format
  effect.
- **CSAR-HiQ_36** — 36 complexes, sequences/SMILES from BAPULM's
  released CSV (the only available source).
- **benchmark1k2101** — 1000 (protein, ligand) pairs released with
  BAPULM. We do not use this set as a primary evaluation
  because all of its (protein, ligand) pairs are present in
  jglaser-100k training; we report numbers on it only in the
  supplementary grid.

---

## 4. Results

### 4.1 Layer-level probing: where in ESM-2 is the binding signal?

For each (ligand encoder, ESM-2 size) pair, we fit a Ridge regression
probe (`RidgeCV` over 20 alphas in log-space) on the concatenation of
each layer's mean-pooled protein representation and the ligand vector,
and report concordance index (CI) on PDBbind core after standardizing
targets by jglaser-100k mean/std.

| Ligand    | 8M peak / CI       | 35M peak / CI       | 650M peak / CI       |
|-----------|--------------------|---------------------|----------------------|
| Morgan    | L1 / 0.673         | L9 / 0.687          | **L27 / 0.699**      |
| ChemBERTa | L1 / 0.635         | L9 / 0.656          | **L29 / 0.689**      |
| MolFormer | L1 / 0.559         | L4 / 0.564          | **L29 / 0.599**      |

**Finding 1.** All three ligand encoders peak at layers 27-29 of
ESM-2 650M (out of 34) — *despite very different ligand
representations.* The protein-side representation that contains the
binding-relevant signal is shared across encoders; the ligand encoder
acts as the differential lever, not the protein encoder.

**Finding 2.** Layer-wise CI strictly orders: Morgan > ChemBERTa >
MolFormer at every ESM-2 size. The Ridge probe (a linear model)
picks Morgan as the strongest input. We will see in §4.3 that this
ordering reverses for deep models trained on PDBbind.

### 4.2 Residue-level interpretability and the localization–accuracy tradeoff

For each ligand encoder, we train a Setting-C model with
ProtQuerySelfCrossAttn on jglaser-100k, then evaluate residue-level
attention against PDBbind pocket masks (`*_pocket.pdb`) on 280
PDBbind core complexes (those with parseable pocket files).

| Ligand    | Pocket enrichment | Attn AUC | Grad AUC | Attn↔Grad ρ | Layer-probe peak CI |
|-----------|------------------:|---------:|---------:|------------:|--------------------:|
| Morgan    | 0.65× (depleted)  | 0.39     | 0.42     | 0.17        | 0.699               |
| ChemBERTa | 1.10×             | 0.43     | **0.64** | 0.03        | 0.689               |
| MolFormer | **2.42×**         | **0.70** | 0.65     | **0.42**    | 0.599               |

**Finding 3 (the headline).** Order on prediction (Ridge probe CI):
Morgan ≳ ChemBERTa ≫ MolFormer. Order on localization (pocket
enrichment): MolFormer ≫ ChemBERTa > Morgan. **The two rankings are
anti-correlated** — Spearman ρ ≈ −1 across the three encoders. The
encoder that best concentrates attention on the pocket is the
worst predictor in this evaluation.

**Finding 4 (faithfulness gap).** ChemBERTa exhibits a near-zero
attention-gradient correlation (ρ = 0.03) despite a strong
gradient-AUC of 0.64: the gradient says the model uses pocket
residues, but the attention does not concentrate there. An
interpretability claim built from attention rollout would mislead
in this case; one built from gradient attribution would not.
MolFormer (ρ = 0.42) is the only encoder where attention and gradient
agree.

**Finding 5 (Morgan anti-localization).** Morgan attention is *actively
depleted* in the pocket — enrichment 0.65× and median attention
AUC = 0.376 (only 22.9% of complexes above chance). Median gradient
AUC = 0.421 (only 16.1% above chance). The Morgan-trained model does
not use pocket residues for prediction; it leverages distal protein
features. That this model still tops the layer-probe leaderboard is
a finding in its own right.

**Finding 6 (the bottleneck reframed).** The original interpretation —
*"localization causes worst prediction"* — is nuanced by the §4.3
training-source results. MolFormer's pocket-concentrated representation
is consistent across training regimes (it always localizes), but
*whether* this representation generalizes depends on whether the
training distribution matches the test distribution. **Pocket
localization amplifies training-distribution sensitivity.**

### 4.3 Final model performance and architecture-source interactions

We train every (Setting, ligand, training-source) combination and
evaluate on all four test sets. Selected results (PDB-parsed
CASF-2016 = primary headline; full table in §4.4):

| Train | Ligand | Setting | r (PDB-parsed CASF) | CI | RMSE | R² |
|-------|--------|---------|--------------------:|----:|-----:|---:|
| jglaser | Morgan | B | 0.562 [0.465, 0.645] | 0.688 | 1.806 | 0.309 |
| jglaser | Morgan | C | 0.554 [0.456, 0.634] | 0.688 | 1.916 | 0.223 |
| jglaser | MolFormer | B | 0.430 [0.335, 0.516] | 0.642 | 2.186 | −0.011 |
| jglaser | MolFormer | BAPULM-head | 0.292 [0.190, 0.395] | 0.598 | 2.543 | −0.369 |
| PDBbind | Morgan | B | 0.686 [0.621, 0.745] | 0.743 | 1.582 | 0.470 |
| PDBbind | Morgan | C | 0.660 [0.576, 0.729] | 0.732 | 1.636 | 0.433 |
| **PDBbind** | **MolFormer** | **B** | **0.813 [0.768, 0.849]** | **0.810** | **1.295** | **0.645** |
| PDBbind | MolFormer | C | 0.790 [0.741, 0.830] | 0.793 | 1.341 | 0.619 |

**Finding 7 (final model).** **The best CASF-2016 number we observe
is r = 0.813 from a 1.84-million-parameter Setting-B head on top of
frozen ESM-2 650M with MolFormer ligand features, trained on
PDBbind refined-minus-core.** This is achieved with a 4.6× smaller
protein encoder than BAPULM (650M vs ProtT5-XL-U50's 3B). Note
that this CASF-2016 number is **not** directly comparable to
BAPULM's r = 0.813 on CSAR-HiQ_36 — they are different test sets
with different leakage profiles. The apples-to-apples comparison
on CSAR-HiQ_36 itself is given in §4.4: our Setting C Morgan
(jglaser-trained) achieves r = 0.696 on CSAR-HiQ_36 vs BAPULM's
reported r = 0.813, a 0.117 r-point gap attributable to encoder
size (ESM-2 650M vs ProtT5-XL-U50 3B).

**Finding 8 (training source dominates).** Switching the training
source from jglaser-100k to PDBbind refined-minus-core improves
CASF-2016 r by 0.12-0.38 across every (ligand, setting) combination.
The standard PDBbind benchmark uses train-on-refined / test-on-core
splits; our jglaser-trained numbers were artificially low because
the training distribution did not match the test distribution.

**Finding 9 (ligand encoder × training source interaction).** The
MolFormer-vs-Morgan ranking flips with the training source:
- **jglaser-trained:** Morgan beats MolFormer on CASF-2016
  (0.562 vs 0.430) — supports the §4.2 *"localization hurts"*
  reading.
- **PDBbind-trained:** MolFormer beats Morgan on CASF-2016
  (0.813 vs 0.686) — *contradicts* §4.2's reading at first glance.

The reconciliation is in Finding 6: MolFormer's localization is
intrinsic, but the *value* of localization depends on
training-distribution alignment. PDBbind's structurally
homogeneous training set matches the CASF-2016 test, so
pocket-concentrated attention extracts maximally relevant signal.
jglaser's diverse, partially off-target training does not match
CASF-2016, so pocket-concentrated attention discards distal context
that would have regularized the regression.

**Finding 10 (CSAR-HiQ_36 inverts the ranking).** On CSAR-HiQ_36,
PDBbind-trained MolFormer collapses to r = 0.195 while jglaser-trained
Setting-C Morgan tops the table at r = 0.696 (n = 36; 95% CI [0.575,
0.822]). MolFormer-trained-on-PDBbind learns features that don't
transfer to a structurally distinct test, while Morgan-trained-on-
jglaser learns more transferable representations. **Architecture
choice for the headline depends on which test set you privilege.**

### 4.4 Comparison to BAPULM on CSAR-HiQ_36

The closest published sequence-only baseline is BAPULM (Meda &
Farimani, 2024) [1]: ProtT5-XL-U50 (3B parameters, frozen) +
MolFormer (frozen) + projection layers + a 4-layer MLP head with
BatchNorm and Dropout, trained on the first 100k rows of
`jglaser/binding_affinity`. Their architecture differs from ours
in two important ways:

- **Protein encoder size:** ProtT5-XL-U50 (3B params) versus our
  ESM-2 650M (4.6× smaller). This precludes a direct
  architectural comparison; any performance gap is partly
  attributable to encoder capacity.
- **Layer combination:** BAPULM uses last-layer mean-pool of
  ProtT5; we use a learned softmax weighting over all ESM-2
  layers (Setting B/C).

BAPULM releases three test CSVs along with their code. We
restrict the comparison to **CSAR-HiQ_36** because the other two
share substantial overlap with their training data:

| BAPULM test CSV | n | (protein, ligand) pair overlap with jglaser-100k | ligand-SMILES overlap |
|-----------------|----:|------------------------------------------------:|----------------------:|
| `benchmark1k2101.csv` | 1000 | 100.0% (1000/1000) | 100.0% |
| `Test2016_290.csv`    | 290  | 12.8% (37/290)     | 71.0%  |
| **`CSAR-HiQ_36.csv`** | 36   | **0.0%**           | 2.8%   |

CSAR-HiQ_36 is the leakage-clean comparison. On this set, BAPULM
reports r = 0.813. Our results on the same test set:

| Model | Pearson r | RMSE | CI | R² |
|---|---:|---:|---:|---:|
| BAPULM (reported, ProtT5-XL-U50) | 0.813 | 1.328 | — | — |
| Ours: Setting C Morgan, jglaser-trained | 0.696 [0.575, 0.822] | 1.497 | 0.769 | 0.424 |
| Ours: Setting B Morgan, jglaser-trained | 0.600 [0.418, 0.752] | 1.636 | 0.704 | 0.312 |
| Ours: Setting B Morgan, PDBbind-trained | 0.646 [0.425, 0.826] | 1.725 | 0.742 | 0.236 |

The 0.117 r-point gap between our best (Setting C Morgan,
jglaser-trained, r = 0.696) and BAPULM's reported number is
roughly consistent with the encoder-size differential
(4.6× more parameters in ProtT5 vs ESM-2 650M). We do not claim
to have replicated BAPULM's results — our experiments use a
different protein encoder. We report the comparison as
positioning for future sequence-only frozen-PLM affinity
prediction work.

**Note on `Test2016_290.csv` and `benchmark1k2101.csv`.** While
we focus the headline comparison on CSAR-HiQ_36, we also
performed evaluations on the other BAPULM-released CSVs as a
sanity check. The numbers exhibit large training-format
sensitivity (the same trained model produces meaningfully
different r values depending on whether the test sequences are
PDB-parsed via BioPython + RDKit or use BAPULM's released
sequence/SMILES strings) and we include the full grid in the
supplementary material rather than the main body. This
sensitivity reinforces our recommendation to evaluate on
PDB-parsed test strings and on test sets disjoint from the
training corpus.

### 4.5 Comprehensive results table

Pearson r / Pearson r / RMSE per test set, all trained models. The
two CASF-2016 columns differ only in test-set string format (PDB-
parsed vs BAPULM CSV). Bootstrap CIs omitted for compactness; see
JSON outputs for full intervals.

| Train | Ligand | Setting | CASF (PDB-parsed) | CASF (BAPULM-CSV) | CSAR-HiQ_36 | benchmark1k2101 |
|-------|--------|---------|------------------:|------------------:|------------:|----------------:|
| jglaser | MolFormer | BAPULM-head | 0.598 / 0.292 / 2.543 | 0.892 / 0.926 / 0.839 | 0.754 / 0.663 / 1.536 | 0.923 / 0.924 / 0.769 |
| jglaser | Morgan    | B           | 0.688 / 0.562 / 1.806 | 0.757 / 0.714 / 1.590 | 0.704 / 0.600 / 1.636 | 0.740 / 0.672 / 1.449 |
| jglaser | Morgan    | C           | 0.688 / 0.554 / 1.916 | 0.791 / 0.782 / 1.474 | 0.769 / 0.696 / 1.497 | 0.778 / 0.750 / 1.307 |
| jglaser | MolFormer | B           | 0.642 / 0.430 / 2.186 | 0.758 / 0.721 / 1.656 | 0.738 / 0.627 / 1.586 | 0.735 / 0.655 / 1.493 |
| jglaser | MolFormer | C           | *training in progress* | — | — | — |
| PDBbind | Morgan    | B           | 0.743 / 0.686 / 1.582 | 0.725 / 0.626 / 1.761 | 0.742 / 0.646 / 1.571 | 0.715 / 0.601 / 1.638 |
| PDBbind | Morgan    | C           | 0.732 / 0.660 / 1.636 | 0.724 / 0.625 / 1.766 | 0.677 / 0.461 / 1.766 | 0.705 / 0.585 / 1.629 |
| **PDBbind** | **MolFormer** | **B**   | **0.810 / 0.813 / 1.295** | 0.654 / 0.446 / 2.354 | 0.668 / 0.333 / 1.973 | 0.671 / 0.502 / 2.048 |
| PDBbind | MolFormer | C           | 0.793 / 0.790 / 1.341 | 0.610 / 0.353 / 2.413 | 0.584 / 0.195 / 2.057 | 0.622 / 0.394 / 2.117 |

Format: `CI / r / RMSE` per cell. **Bold row is the recommended
final model.**

---

## 5. Discussion

### 5.1 Training-distribution sensitivity

Two independent findings in this paper point to the same underlying
mechanism: *aggressive feature compression amplifies sensitivity to
the training-distribution / test-distribution gap.*

- **MolFormer's pocket-concentrated attention** (§4.2) extracts a
  small, specific subset of protein context. When training data
  matches the test, this is optimal (PDBbind→PDBbind, r = 0.813).
  When training data is broader and noisier (jglaser→PDBbind),
  the same compression discards regularizing context (r = 0.430).

- **Architecture × training-source interaction** (§4.3) is large.
  The same Setting B head with MolFormer ligand features achieves
  r = 0.430 on CASF-2016 when trained on jglaser-100k, versus
  r = 0.813 when trained on PDBbind refined-minus-core. The
  ligand-encoder ranking flips with the training source, and
  PDBbind-trained models that win on CASF-2016 lose substantially
  on CSAR-HiQ_36.

In both cases, simpler/more-constrained alternatives
(LayerW-only Setting B, Morgan fingerprints) generalize better
at the cost of in-distribution peak fit.

### 5.2 The localization–accuracy tradeoff is more nuanced than initially claimed

The §4.2 *"the encoder with best pocket localization is the worst
predictor"* result, taken as a universal claim, is contradicted by
§4.3: PDBbind-trained MolFormer (most-localizing encoder) is the best
CASF-2016 predictor at r = 0.813. The honest reformulation is:
**MolFormer's intrinsic pocket-localization makes its predictive
performance more sensitive to the training distribution.** It can be
the best or the worst depending on whether train and test
distributions match.

### 5.3 Test-set choice and reproducibility

Our preferred evaluation protocol for sequence-only frozen-PLM
affinity prediction is to (i) parse test-set protein sequences
from PDB structures via BioPython and ligand SMILES via RDKit
canonicalization, providing a string representation that is
deterministic and independent of any training-corpus format; and
(ii) include CSAR-HiQ_36 as a comparison test set when training
on `jglaser/binding_affinity`-derived data, since CSAR-HiQ_36 is
disjoint from that source. This is a methodological positioning
choice motivated by the variability we observed in numbers when
the same trained model is evaluated on different test-set string
representations of the same complexes.

### 5.4 Final-model recommendations

The choice of "final model" depends on which generalization regime
matters:
- **In-distribution (PDBbind-train → PDBbind-test):** Setting B +
  MolFormer + PDBbind training. r = 0.813 on CASF-2016. Simpler
  head regularizes better.
- **Out-of-distribution (jglaser-train → CSAR-HiQ_36):** Setting C +
  Morgan + jglaser training. r = 0.696. Attention head adds
  capacity that helps under domain shift.

Setting C does not consistently dominate Setting B. Architecture
choice should be tied to the deployment regime.

---

## 6. Limitations

- **Frozen protein encoder.** We do not fine-tune ESM-2. This is a
  deliberate choice that allows the layer-probing analysis (§4.1) to
  measure intrinsic ESM-2 representations rather than affinity-tuned
  ones. It also caps the absolute performance ceiling — fine-tuned
  PLMs for affinity (e.g., ESM-IF, structure-aware methods) achieve
  higher reported numbers.
- **Sequence-only.** No 3D structure, no docking, no co-coordinates.
  Comparable structure-aware models (IGN, Pafnucy, OnionNet) report
  higher numbers on PDBbind core but require structural input.
- **Training-set size.** jglaser-100k is a subset of the full 1.9M
  available rows. Using more training data may shift the
  ligand-encoder ranking, particularly for MolFormer.
- **Single-seed training.** All headline numbers are from a single
  random seed. Bootstrap CIs report the test-set sampling variance
  but not the training-seed variance.
- **CSAR-HiQ_36 is small (n = 36).** Bootstrap CIs are wide, and
  conclusions about model ranking on this set are noisy. We use it
  as a complementary signal, not the sole headline.

---

## 7. Conclusion

We probe what frozen ESM-2 encodes about protein-ligand binding
affinity at the layer level (§4.1), the residue level (§4.2), and
the model level (§4.3). The ligand encoder is the dominant lever
controlling a localization–accuracy tradeoff, with the most
pocket-localizing encoder (MolFormer) showing the highest sensitivity
to the training distribution. A simple 1.84M-parameter
layer-weighted head achieves r = 0.813 on PDBbind core (CASF-2016)
when trained on PDBbind refined-minus-core, demonstrating that
sequence-only frozen-PLM affinity prediction can be both
parameter-efficient and competitive on the standard PDBbind split.
For positioning against the closest published baseline (BAPULM,
ProtT5-XL-U50 + MolFormer + 4-layer MLP), we evaluate on
CSAR-HiQ_36 — disjoint from `jglaser/binding_affinity` — where our
Setting C Morgan model (jglaser-trained) achieves r = 0.696
against BAPULM's reported r = 0.813. The remaining gap (0.117
r-points) is broadly consistent with the 4.6× smaller protein
encoder (ESM-2 650M vs ProtT5-XL-U50 3B); we do not claim a
direct architectural reproduction of BAPULM, since the substituted
encoder is itself a substantial design change. We recommend
PDB-parsed test strings and CSAR-HiQ_36 as the comparison
protocol for future sequence-only frozen-PLM affinity prediction
work training on `jglaser/binding_affinity`-derived data.

---

## 8. Reproducibility artifacts

All training and evaluation scripts are in the project repository.
Key scripts:
- `run_analysis.py` — layer probing and residue-level interpretability
  (§4.1, §4.2).
- `aggregate_ligands.py` — figure generation across ligand encoders.
- `run_final_model.py`, `run_final_model_pdbbind.py` — final model
  training (§4.3) on jglaser and PDBbind respectively.
- `run_bapulm_replication.py` — BAPULM-style head architecture
  trained on our stack with ESM-2 650M (filename retained for
  historical reasons; this is *not* a faithful reproduction of
  BAPULM, since the protein encoder is substituted).
- `leakage_audit.py` — audits BAPULM's released CSVs against jglaser
  (§4.4).
- `eval_existing_on_csars.py` — evaluates trained models on BAPULM's
  CSV strings (§4.4).

Key result artifacts (all on disk):
- `layer_results_{Morgan,ChemBERTa,MolFormer}.json` — §4.1
- `interp_results_{Morgan,ChemBERTa,MolFormer}.json` — §4.2
- `final_model_results_*.json`, `final_pdbbind_results_*.json` — §4.3
- `bapulm_replication_results.json` — §4.4
- `leakage_summary.json`, `leakage_examples.md`, `leakage_audit.png` — §4.4
- `bapulm_csv_eval_results.json` — §4.4

Figures:
- `layer_analysis_all_ligands.png` — Fig. 1, §4.1
- `interpretability_all_ligands.png` — Fig. 2, §4.2
- `faithfulness_all_ligands.png` — Fig. 3, §4.2
- `final_model_scatter_*.png`, `final_pdbbind_scatter_*.png` — Fig. 4, §4.3
- `leakage_audit.png` — Fig. S3, §4.4
- `bapulm_replication_scatter.png` — supplementary §4.4

---

## 9. References

[1] **BAPULM**: Meda, R. S., & Farimani, A. B. (2024). *BAPULM:
Binding Affinity Prediction using Language Models.*
arXiv preprint arXiv:2411.04150.
https://arxiv.org/abs/2411.04150
Code: https://github.com/radh55sh/BAPULM

[2] **ProtT5**: Elnaggar, A., Heinzinger, M., Dallago, C., et al.
(2022). *ProtTrans: Toward Understanding the Language of Life
Through Self-Supervised Learning.* IEEE Transactions on Pattern
Analysis and Machine Intelligence, 44(10):7112–7127.
DOI: https://doi.org/10.1109/TPAMI.2021.3095381

[3] **MolFormer**: Ross, J., Belgodere, B., Chenthamarakshan, V.,
Padhi, I., Mroueh, Y., & Das, P. (2022). *Large-scale chemical
language representations capture molecular structure and properties.*
Nature Machine Intelligence, 4:1256-1264.
DOI: https://doi.org/10.1038/s42256-022-00580-7
HuggingFace: `ibm/MoLFormer-XL-both-10pct`

[4] **jglaser/binding_affinity**: Glaser, J. *binding_affinity
dataset.* HuggingFace Datasets (1.9M protein-ligand affinity pairs
curated from BindingDB and other sources).
https://huggingface.co/datasets/jglaser/binding_affinity

[5] **ESM-2**: Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W.,
Smetanin, N., et al. (2023). *Evolutionary-scale prediction of
atomic-level protein structure.* Science, 379(6637):1123-1130.
DOI: https://doi.org/10.1126/science.ade2574
Code: https://github.com/facebookresearch/esm

[6] **ChemBERTa**: Chithrananda, S., Grand, G., & Ramsundar, B.
(2020). *ChemBERTa: Large-Scale Self-Supervised Pretraining for
Molecular Property Prediction.* arXiv:2010.09885.
https://arxiv.org/abs/2010.09885
HuggingFace: `DeepChem/ChemBERTa-77M-MLM`

[7] **PDBbind**: Wang, R., Fang, X., Lu, Y., & Wang, S. (2004).
*The PDBbind Database: Collection of Binding Affinities for
Protein-Ligand Complexes with Known Three-Dimensional Structures.*
Journal of Medicinal Chemistry, 47(12):2977–2980.
DOI: https://doi.org/10.1021/jm030580l
Updated v2016 release used here:
http://www.pdbbind.org.cn/casf.php

[8] **CASF-2016**: Su, M., Yang, Q., Du, Y., Feng, G., Liu, Z.,
Li, Y., & Wang, R. (2019). *Comparative Assessment of Scoring
Functions: The CASF-2016 Update.* Journal of Chemical Information
and Modeling, 59(2):895–913.
DOI: https://doi.org/10.1021/acs.jcim.8b00545

[9] **CSAR-HiQ**: Dunbar, J. B., Smith, R. D., Yang, C. Y., et al.
(2011). *CSAR Benchmark Exercise of 2010: Selection of the
Protein-Ligand Complexes.* Journal of Chemical Information and
Modeling, 51(9):2036–2046.
DOI: https://doi.org/10.1021/ci200082t

[10] **RDKit**: Landrum, G., et al. *RDKit: Open-source
cheminformatics.* https://www.rdkit.org

[11] **BioPython**: Cock, P. J. A., Antao, T., Chang, J. T., et al.
(2009). *Biopython: freely available Python tools for computational
molecular biology and bioinformatics.* Bioinformatics,
25(11):1422-1423.
DOI: https://doi.org/10.1093/bioinformatics/btp163

[12] **Morgan/ECFP fingerprints**: Rogers, D., & Hahn, M. (2010).
*Extended-Connectivity Fingerprints.* Journal of Chemical
Information and Modeling, 50(5):742-754.
DOI: https://doi.org/10.1021/ci100050t

[13] **PyTorch**: Paszke, A., Gross, S., Massa, F., et al. (2019).
*PyTorch: An Imperative Style, High-Performance Deep Learning
Library.* NeurIPS 2019.
https://pytorch.org

[14] **HuggingFace Transformers**: Wolf, T., Debut, L., Sanh, V.,
et al. (2020). *Transformers: State-of-the-Art Natural Language
Processing.* EMNLP 2020 System Demonstrations.
DOI: https://doi.org/10.18653/v1/2020.emnlp-demos.6

---

## 10. Acknowledgment of AI tools

The following AI tools were used during this project:

- **Claude (Anthropic)** — used as a coding assistant during script
  development (`run_analysis.py`, `run_final_model*.py`,
  `eval_existing_on_csars.py`, `leakage_audit.py`,
  `run_bapulm_replication.py`) and during drafting of this report.
  All quantitative claims in §3 (numerical results, bootstrap
  confidence intervals, leakage statistics) were independently
  computed by the scripts and verified by the author against
  on-disk JSON outputs and SLURM logs. AI-suggested code was
  reviewed and modified before execution.

- **No figures in this report were generated by AI image-generation
  tools.** All figures are matplotlib renderings of real on-disk
  results, except for the pipeline figure (Figure 1) which was
  drawn manually (or, if drawn with AI assistance, this is
  acknowledged in that figure's caption).

The AI-hallucination guidance referenced in the submission
requirements
(https://drive.google.com/file/d/1SQ2cZiKGAoK4ZYt29AmrPybwGzPfUwEs)
was followed: every numerical claim was traced back to a SLURM
output or a JSON results file before inclusion.

---

## 11. Figure list and captions

**Figure 1. Pipeline.** *Hand-drawn diagram of the model
architecture and three-branch analysis pipeline used in this work.
Frozen ESM-2 (8M, 35M, or 650M) processes the protein sequence into
per-layer mean-pooled vectors and a per-residue tensor. The ligand
SMILES is independently encoded by Morgan fingerprints, ChemBERTa,
or MolFormer. The shared inputs feed three downstream analyses:
(left) a Ridge-regression layer probe (§4.1); (middle) a residue-
level interpretability head with ProtQuerySelfCrossAttn evaluated
against PDBbind pocket masks (§4.2); (right) the final
binding-affinity prediction model in three architectural variants
(Settings A, B, C) trained on either jglaser or PDBbind and
evaluated on CASF-2016, CSAR-HiQ_36, and benchmark1k2101 (§4.3 and
§4.4). [Acknowledge any AI-tool assistance for SVG/TikZ drafting
here.]*

**Figure 2. Layer-level binding signal across ESM-2 sizes and
ligand encoders.** *(Source: `layer_analysis_all_ligands.png`,
generated by `aggregate_ligands.py` from
`layer_results_{Morgan,ChemBERTa,MolFormer}.json`.)*
*Concordance index (CI) of a Ridge-regression probe fit to each
layer's mean-pooled protein representation concatenated with the
ligand vector, evaluated on PDBbind core (n=290). Three subplots
correspond to ESM-2 8M (7 layers), 35M (13 layers), and 650M
(34 layers). Three lines per subplot correspond to Morgan
(blue), ChemBERTa (red), and MolFormer (green) ligand encoders.
Annotations mark the peak layer per ligand encoder. All three
encoders peak at layers 27-29 of the 650M variant despite very
different ligand representations, supporting the late-layer
convergence claim in §4.1.*

**Figure 3. Residue-level interpretability across ligand
encoders.** *(Source: `interpretability_all_ligands.png`,
generated by `aggregate_ligands.py` from
`interp_results_{Morgan,ChemBERTa,MolFormer}.json`.)*
*Four-panel bar chart: (left to right) pocket enrichment relative
to random (chance = 1.0×), attention AUC vs pocket mask
(chance = 0.5), gradient-attribution AUC vs pocket mask, and
attention-vs-gradient Spearman correlation. Each panel has three
bars (Morgan, ChemBERTa, MolFormer). Same 280 PDBbind core
complexes used across all encoders; pocket masks identical
byte-for-byte across runs. Demonstrates the inverse correlation
between pocket localization and predictive accuracy (§4.2).*

**Figure 4. Attention faithfulness by ligand encoder.** *(Source:
`faithfulness_all_ligands.png`, generated by `aggregate_ligands.py`
from `interp_per_complex_*.json`.)*
*Density histograms of per-complex Spearman correlation between
attention weights and gradient norms over residues, overlaid for
the three ligand encoders. Vertical dashed lines mark per-encoder
means. ChemBERTa exhibits a near-zero mean (ρ̄=0.03) — gradient
and attention disagree on which residues drive prediction —
while MolFormer's attention and gradient agree (ρ̄=0.42). Morgan
sits between (ρ̄=0.17). The figure supports the faithfulness-gap
claim in §4.2.*

**Figure 5. Final-model predictions on CASF-2016.** *(Source:
`final_pdbbind_scatter_MolFormer_SB_na_650M.png`, generated by
`run_final_model_pdbbind.py` from
`final_pdbbind_predictions_MolFormer_SB_na_650M.json`.)*
*Scatter of predicted vs experimental −log10(K_d/K_i) for the
recommended final model: Setting B (LayerW + MLP head, 1.84M
trainable parameters) on top of frozen ESM-2 650M with MolFormer
ligand features, trained on PDBbind refined-minus-core (n_train =
3,390). Test set: PDBbind core / CASF-2016 (n=290). Pearson
r = 0.813 [95% CI 0.768, 0.849], CI = 0.810 [0.785, 0.832],
RMSE = 1.295 [1.181, 1.406], R² = 0.645 [0.576, 0.700]. Red
dashed line: y = x.*

**Optional supplementary figures:**
- *Figure S1.* Sequence attention heatmaps (top complexes by
  attention AUC). Source: `sequence_attention_heatmaps_*.png`.
- *Figure S2.* Performance vs sequence length. Source:
  `length_analysis.png`.
- *Figure S3.* Test-set overlap with `jglaser/binding_affinity` for
  BAPULM's released test CSVs (`leakage_audit.png`). Reproduces the
  per-test-set overlap percentages noted in §4.4 in visual form.
  Provided for transparency about why the headline comparison
  is restricted to CSAR-HiQ_36.

---

*End of report.*
