RaRF baseline (alias-tolerant) — Quick Start

Files
- run_baseline_userutils.py — drop-in script that works with your current Excel sheet names (df, imine, nuc, ligand, solvent). Requires utils.py in the same folder.
- Nature_SMILES.xlsx — your data workbook (not included here).
- Outputs: preds.csv, coverage.csv, summary.json, usage_edges.csv, usage_matrix.csv

Assumptions
- Your conda env has scikit-learn, pandas, numpy, RDKit (used inside utils.py).
- utils.py exposes: smi2morgan(smiles, nbits) and get_distances(X_train, X_test) returning a distance matrix.

Run
(rarfviz) python run_baseline_userutils.py

Options (edit inside the script)
- EXCEL_PATH: path to workbook (default "Nature_SMILES.xlsx")
- REACTIONS_SHEET: set to a specific sheet if not "df" or auto-detectable
- TARGET_COL: default "DDG"
- RADIUS: Jaccard distance cutoff (default 0.45). Fallback loosens once to 0.60.
- N_BITS_PER_PART: Morgan FP size per reaction part (default 2048)
- N_EST: RandomForest trees (default 300)

Notes
- The script accepts either four *_SMILES columns in the reactions sheet OR name columns resolved via lookup sheets. Both column and sheet names are case-insensitive; common aliases like "nuc" are supported.
- If no neighbors fall within the radius, it automatically loosens once, then falls back to the nearest neighbor.
