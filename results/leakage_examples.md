# Concrete leakage examples

Each row below is a (protein, ligand) pair that appears **identically** in BAPULM's published test CSV and in the first 100k rows of `jglaser/binding_affinity` (BAPULM's training data, per their `main.py`/`config.yaml`).

Source: cloned from `github.com/radh55sh/BAPULM` on 2026-05-03.

## benchmark1k2101  —  1000/1000 pairs leaked (100.0%)

| pdbid | test affinity | train affinity (jglaser) | seq len | smiles |
|:-----|--------------:|-------------------------:|--------:|:-------|
| ? | 9.155 | 9.155 | 110 | `COc1cc2c3c(Nc4nn(C)c5ccccc45)nc(C)[nH]c-3nc2cc1-c1c(C)noc1C` |
| ? | 7.854 | 7.854 | 264 | `COc1cc(-c2cnc3[nH]cc(C(=O)NC(C)C)c3n2)cc(OC)c1OC` |
| ? | 6.678 | 6.678 | 271 | `c1ccc(-c2oc3nccc(NCC[NH+]4CC[NH2+]CC4)c3c2-c2ccccc2)cc1` |
| ? | 9.337 | 9.337 | 99 | `COc1ccc(S(=O)(=O)N(C[C@H]2CCC(=O)N2)C[C@@H](O)[C@H](Cc2ccccc2)NC(=O)O[C@@H]2C[C@@H]3CCO[C@@H]3C2)cc1` |
| ? | 8.097 | 8.097 | 99 | `CC(C)C1NC(=O)[C@@H](C(O)[C@H](Cc2ccccc2)NC(=O)[C@@H](NC(=O)OCc2ccccc2)C(C)(C)C)NCc2ccc(cc2)OCCOCCNC1=O` |

## Test2016_290  —  37/290 pairs leaked (12.8%)

| pdbid | test affinity | train affinity (jglaser) | seq len | smiles |
|:-----|--------------:|-------------------------:|--------:|:-------|
| 3gv9 | 2.120 | 2.125 | 358 | `CC(=O)Nc1ccsc1C(=O)O` |
| 3gr2 | 2.520 | 2.523 | 358 | `CC[C@H]1C(=O)N(c2nnn[nH]2)N=C1C` |
| 3fcq | 2.770 | 2.770 | 316 | `CC(=O)Oc1c(C)cccc1C(=O)O` |
| 3lka | 2.820 | 2.824 | 158 | `COc1ccc(S(N)(=O)=O)cc1` |
| 4kz6 | 3.100 | 3.097 | 358 | `C[C@@H]1CCC[C@H](C(=O)O)N1C(=O)CCS` |

## CSAR-HiQ_36  —  0/36 pairs leaked (0.0%)

| pdbid | test affinity | train affinity (jglaser) | seq len | smiles |
|:-----|--------------:|-------------------------:|--------:|:-------|
| — | — | — | — | (no exact pair matches) |
