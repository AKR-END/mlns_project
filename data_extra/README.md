# `data_extra/` — comparison test sets from BAPULM

The CSV files in this directory are **mirrored from the BAPULM
repository** for reproducibility of the comparison study described in
§4.4 of the project report.

| File | Source | Purpose in this project |
|------|--------|-------------------------|
| `Test2016_290.csv` | github.com/radh55sh/BAPULM | The CASF-2016 / PDBbind-core 290 complexes in the sequence/SMILES format BAPULM uses for evaluation. We use this to characterize how test-set string format affects measured performance. |
| `CSAR-HiQ_36.csv`  | github.com/radh55sh/BAPULM | 36 protein-ligand complexes from the CSAR-HiQ benchmark. This is the test set used for the BAPULM comparison in our report (§4.4). |
| `benchmark1k2101.csv` | github.com/radh55sh/BAPULM | 1000 protein-ligand pairs from BindingDB. We document its substantial overlap with `jglaser/binding_affinity` training data in §4.4 and recommend not using it as a held-out evaluation when training on jglaser. |

Cited as: Meda, R. S., & Farimani, A. B. (2024). *BAPULM: Binding
Affinity Prediction using Language Models.* arXiv:2411.04150.
https://github.com/radh55sh/BAPULM

The CSVs are included unchanged from the upstream BAPULM repository.
PDBbind-core complexes are also independently parsed from the
PDBbind v2016 release (`load_pdbbind_set("core")` in `utils.py`)
via BioPython + RDKit; we refer to those as **PDB-parsed strings** in
the report and recommend them as the canonical test-set
representation.
