#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent RaRF baseline (uses user's utils.py: smi2morgan, get_distances).
Tolerant to lowercase / alias sheet names found in Nature_SMILES.xlsx:
  - reactions sheet defaults to 'df' (case-insensitive) unless REACTIONS_SHEET is set
  - lookup sheets may be named: imine/Imine, nuc/Nucleophile, ligand/Ligand, solvent/Solvent
  - accepts either *_SMILES columns (any case) OR name columns mapped via lookup sheets

Outputs:
  - preds.csv, coverage.csv, summary.json, usage_edges.csv, usage_matrix.csv
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- user utils (RDKit + distances) ---
from utils import smi2morgan, get_distances

# ----------------------------
# Config (edit if you like)
# ----------------------------
EXCEL_PATH      = "Nature_SMILES.xlsx"
REACTIONS_SHEET = None          # e.g., "df"; if None, auto-detect a sheet with TARGET_COL
TARGET_COL      = "DDG"
N_BITS_PER_PART = 2048
RADIUS          = 0.45          # Jaccard distance radius; neighbors have D <= RADIUS
TAU             = 1.0 - RADIUS  # similarity threshold (S = 1 - D)
RADIUS_FALLBACK = 0.60          # if no neighbors inside radius, use this looser radius (distance)
N_EST           = 300
TEST_SIZE       = 0.20
SEED            = 42

# outputs
OUT_PRED   = "preds.csv"
OUT_COV    = "coverage.csv"
OUT_SUM    = "summary.json"
OUT_EDGES  = "usage_edges.csv"
OUT_MATRIX = "usage_matrix.csv"

# ----------------------------
# Helpers for robust IO
# ----------------------------
def _lower_map(items):
    """Map lowercase -> original for a list of strings."""
    return {str(x).lower(): x for x in items}

def _resolve_sheet(xl, aliases):
    """
    Resolve a sheet name in a pandas.ExcelFile given a list of alias candidates.
    Case-insensitive.
    """
    name_map = _lower_map(xl.sheet_names)
    for cand in aliases:
        key = str(cand).lower()
        if key in name_map:
            return name_map[key]
    raise KeyError(f"None of the candidate sheets found: {aliases}. Available: {xl.sheet_names}")

def _resolve_col(df, *cands):
    """
    Return real column name in df that matches any candidate (case-insensitive).
    Raises KeyError if none found.
    """
    cmap = _lower_map(df.columns)
    for c in cands:
        if str(c).lower() in cmap:
            return cmap[str(c).lower()]
    raise KeyError(f"Missing required column. Tried {cands}. Found: {list(df.columns)}")

def _has_cols(df, cols):
    cmap = {c.lower() for c in df.columns}
    return all(c.lower() in cmap for c in cols)

def build_X_from_excel_local(path, reactions_sheet=None, target_col="DDG", nbits=2048):
    xl = pd.ExcelFile(path)

    # --- pick reactions sheet
    if reactions_sheet is not None:
        rxn_sheet = _resolve_sheet(xl, [reactions_sheet])
    else:
        # prefer 'df', else first sheet containing target_col
        try:
            rxn_sheet = _resolve_sheet(xl, ["df"])
        except KeyError:
            rxn_sheet = None
            for s in xl.sheet_names:
                tmp = xl.parse(s)
                if target_col in tmp.columns or target_col.lower() in [c.lower() for c in tmp.columns]:
                    rxn_sheet = s
                    break
            if rxn_sheet is None:
                raise ValueError(f"Could not find a reactions sheet with column '{target_col}'. Available: {xl.sheet_names}")

    rxn = xl.parse(rxn_sheet)

    # Normalize needed column handles (case-insensitive)
    target_col_real = _resolve_col(rxn, target_col, target_col.lower())

    # Option A: *_SMILES columns directly in reactions sheet (case-insensitive)
    smiles_candidates = [
        ("Imine_SMILES", "imine_smiles"),
        ("Nucleophile_SMILES", "nucleophile_smiles", "nuc_smiles"),
        ("Ligand_SMILES", "ligand_smiles"),
        ("Solvent_SMILES", "solvent_smiles"),
    ]
    smiles_cols_real = []
    ok_smiles = True
    for cands in smiles_candidates:
        try:
            smiles_cols_real.append(_resolve_col(rxn, *cands))
        except KeyError:
            ok_smiles = False
            smiles_cols_real.append(None)

    if ok_smiles:
        parts = [rxn[c] for c in smiles_cols_real]
        y = rxn[target_col_real].to_numpy(dtype=float)
        X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)
        for i in range(len(rxn)):
            bits = [smi2morgan(str(parts[j].iloc[i]), nbits=nbits) for j in range(4)]
            X[i] = np.concatenate(bits, axis=0)
        return X, y

    # Option B: name columns + lookup sheets (tolerant to lowercase/aliases)
    # name columns in reactions
    name_cols_spec = {
        "Imine": ("Imine", "imine"),
        "Nucleophile": ("Nucleophile", "nucleophile", "nuc"),
        "Ligand": ("Ligand", "ligand"),
        "Solvent": ("Solvent", "solvent"),
    }
    name_cols_real = {}
    for key, cands in name_cols_spec.items():
        name_cols_real[key] = _resolve_col(rxn, *cands)

    # lookup sheets + Name/SMILES columns
    lookup_aliases = {
        "Imine": ["Imine", "imine"],
        "Nucleophile": ["Nucleophile", "nucleophile", "nuc"],
        "Ligand": ["Ligand", "ligand"],
        "Solvent": ["Solvent", "solvent"],
    }
    lookups = {}
    for key, aliases in lookup_aliases.items():
        sname = _resolve_sheet(xl, aliases)
        df = xl.parse(sname)
        name_col = _resolve_col(df, "Name", "name")
        smi_col  = _resolve_col(df, "SMILES", "smiles")
        lookups[key] = dict(zip(df[name_col].astype(str), df[smi_col].astype(str)))

    # build X from lookup
    y = rxn[target_col_real].to_numpy(dtype=float)
    X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)

    missing = []
    for i, row in rxn.iterrows():
        names = [
            str(row[name_cols_real["Imine"]]),
            str(row[name_cols_real["Nucleophile"]]),
            str(row[name_cols_real["Ligand"]]),
            str(row[name_cols_real["Solvent"]]),
        ]
        smis = [
            lookups["Imine"].get(names[0], ""),
            lookups["Nucleophile"].get(names[1], ""),
            lookups["Ligand"].get(names[2], ""),
            lookups["Solvent"].get(names[3], ""),
        ]
        # track missing
        for k, nm, sm in zip(["Imine","Nucleophile","Ligand","Solvent"], names, smis):
            if not sm:
                missing.append((i, k, nm))
        bits = [smi2morgan(s, nbits=nbits) for s in smis]
        X[i] = np.concatenate(bits, axis=0)

    if missing:
        # show a compact preview; do not hard fail, but warn (NaN predictions will surface later)
        uniq = {}
        for i, k, nm in missing:
            uniq.setdefault(k, set()).add(nm)
        msg = "WARNING: Missing SMILES for some names →\n" + "\n".join(
            f"  {k}: {sorted(list(v))[:10]}{' ...' if len(v)>10 else ''}" for k, v in uniq.items()
        )
        print(msg)

    return X, y

def main():
    t0 = time.time()
    print("="*80)
    print("🔷 BASELINE (user utils): Independent RaRF using utils.get_distances")
    print("="*80)

    print("\n[1] Loading & fingerprinting ...")
    X, y = build_X_from_excel_local(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    print(f"    X: {X.shape} | y: {y.shape} | bits/part={N_BITS_PER_PART} | sheet={REACTIONS_SHEET or 'auto'}")

    print("\n[2] Train/test split (80/20, seed=42) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    print(f"    X_tr: {X_tr.shape} | X_te: {X_te.shape}")

    print("\n[3] Jaccard distances (test vs train) via utils.get_distances ...")
    # user utils signature: get_distances(X_train, X_test) -> (n_test x n_train) distances
    D = get_distances(X_tr, X_te)
    if D.shape != (X_te.shape[0], X_tr.shape[0]):
        # try transposing if user implementation differs
        if D.shape == (X_tr.shape[0], X_te.shape[0]):
            D = D.T
        else:
            raise RuntimeError(f"Distance matrix unexpected shape {D.shape}; expected {(X_te.shape[0], X_tr.shape[0])}")
    S = 1.0 - D
    tau = TAU
    print(f"    radius={RADIUS:.2f} → tau (similarity)={tau:.2f}")

    print("\n[4] Per-target neighbor selection & prediction (RaRF = all in-radius) ...")
    rf_params = dict(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    n_test, n_train = X_te.shape[0], X_tr.shape[0]
    y_pred = np.full(n_test, np.nan)
    neigh_lists = []
    neigh_counts = []
    usage_edges = []                 # (target_i, train_j, rank)
    usage_matrix = np.zeros((n_test, n_train), dtype=np.uint8)

    for i in range(n_test):
        # inside radius in similarity space: S >= tau  (equivalently D <= RADIUS)
        inrad = np.where(S[i] >= tau)[0]
        # if empty, loosen radius once (fallback)
        if inrad.size == 0:
            fallback_tau = 1.0 - RADIUS_FALLBACK
            inrad = np.where(S[i] >= fallback_tau)[0]
        # if still empty, take the single nearest neighbor
        if inrad.size == 0:
            inrad = np.array([int(np.argmax(S[i]))], dtype=int)

        # sort by similarity (desc)
        order = np.argsort(S[i, inrad])[::-1]
        use_idx = inrad[order]

        neigh_lists.append(use_idx.tolist())
        neigh_counts.append(len(use_idx))
        for r, tr_idx in enumerate(use_idx):
            usage_edges.append((i, int(tr_idx), r))
            usage_matrix[i, tr_idx] = 1

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr[use_idx], y_tr[use_idx])
        y_pred[i] = rf.predict(X_te[i:i+1])[0]

    print("    Done. NaN preds:", int(np.isnan(y_pred).sum()))

    print("\n[5] Metrics & cost accounting ...")
    mask = ~np.isnan(y_pred)
    mae  = float(mean_absolute_error(y_te[mask], y_pred[mask])) if mask.any() else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_te[mask], y_pred[mask]))) if mask.any() else float("nan")
    r2   = float(r2_score(y_te[mask], y_pred[mask])) if mask.any() else float("nan")

    per_target_cost = int(sum(len(lst) for lst in neigh_lists))        # NO dedup
    unique_cost     = int(usage_matrix.sum(axis=0).astype(bool).sum()) # Deduped union

    print(f"    MAE : {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R^2 : {r2:.4f}")
    print(f"    Per-target cost (no-dedup): {per_target_cost}")
    print(f"    Unique-experiment cost (deduped union): {unique_cost}")

    # [6] Save artifacts
    n_test_int = int(n_test)
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred,
        "abs_err": np.abs(y_te - y_pred),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    }).to_csv(OUT_PRED, index=False); print("    →", OUT_PRED)

    pd.DataFrame({
        "target_idx": np.arange(n_test_int, dtype=int),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    }).to_csv(OUT_COV, index=False); print("    →", OUT_COV)

    pd.DataFrame(usage_edges, columns=["target_idx", "train_idx", "rank"]).to_csv(OUT_EDGES, index=False); print("    →", OUT_EDGES)

    matrix_cols = [f"train_{j}" for j in range(n_train)]
    mat_df = pd.DataFrame(usage_matrix, columns=matrix_cols)
    mat_df.insert(0, "target_idx", np.arange(n_test_int, dtype=int))
    mat_df.to_csv(OUT_MATRIX, index=False); print("    →", OUT_MATRIX)

    with open(OUT_SUM, "w") as f:
        json.dump(dict(
            excel=EXCEL_PATH, sheet=REACTIONS_SHEET or "auto", target_col=TARGET_COL,
            radius=RADIUS, tau=tau,
            bits_per_part=N_BITS_PER_PART,
            rf_estimators=N_EST, seed=SEED, test_size=TEST_SIZE,
            n_test=n_test_int,
            mae=mae, rmse=rmse, r2=r2,
            per_target_cost=per_target_cost,
            unique_experiment_cost=unique_cost,
            runtime_sec=round(time.time() - t0, 3),
            outputs=[OUT_PRED, OUT_COV, OUT_SUM, OUT_EDGES, OUT_MATRIX],
        ), f, indent=2); print("    →", OUT_SUM)

    print("\nDONE.")

if __name__ == "__main__":
    main()
