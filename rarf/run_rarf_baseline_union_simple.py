#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Super-simple RaRF baseline (no CLI args).
- Loads Nature_SMILES.xlsx (sheet="Reactions", target column "DDG").
- Builds ECFP4 fingerprints (tries utils.build_X_from_excel; else expects *_SMILES columns).
- Trains independent per-target RaRF local models.
- Reports MAE/RMSE/R2.
- Logs per-target neighbor lists, unique-experiment cost (deduped union), per-target cost (sum).
- Writes:
    baseline_rarf_baseline_predictions.csv
    baseline_rarf_baseline_neighbors.json
    baseline_rarf_baseline_summary.json
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# ----------------------------
# Fingerprinting helpers
# ----------------------------
def _rdkit_ecfp4_bits(smiles, n_bits=2048):
    """Return numpy array of 0/1 bits for an RDKit ECFP4 fingerprint."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except Exception as e:
        raise ImportError("RDKit is required for fingerprinting. Install rdkit-pypi.") from e

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def _concat_bits(parts, n_bits=2048):
    """Concatenate bit vectors for each reaction component."""
    return np.concatenate([_rdkit_ecfp4_bits(smi, n_bits) for smi in parts], axis=0)

def _build_X_from_smiles_df(df, n_bits=2048):
    """Fallback builder: expects *_SMILES columns present."""
    needed = ["Imine_SMILES", "Nucleophile_SMILES", "Ligand_SMILES", "Solvent_SMILES"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Fallback builder expected column '{col}' but it is missing. "
                           f"If your Excel uses name→SMILES lookups, please ensure utils.build_X_from_excel is available.")
    X = []
    for _, row in df.iterrows():
        parts = [row["Imine_SMILES"], row["Nucleophile_SMILES"], row["Ligand_SMILES"], row["Solvent_SMILES"]]
        X.append(_concat_bits(parts, n_bits=n_bits))
    X = np.vstack(X).astype(np.uint8)
    return X

def load_features(excel_path="Nature_SMILES.xlsx", sheet="Reactions", target_col="DDG", n_bits=2048, test_size=0.2, seed=42):
    """
    Try to reuse earlier Excel→fingerprint builder:
      from utils import build_X_from_excel
    Fallback to *_SMILES columns if present.
    """
    try:
        from utils import build_X_from_excel  # your earlier helper (if present)
        df = pd.read_excel(excel_path, sheet_name=sheet)
        X, y = build_X_from_excel(excel_path, reactions_sheet=sheet, target_col=target_col, nbits=n_bits)
    except Exception:
        # Fallback: simple SMILES columns
        df = pd.read_excel(excel_path, sheet_name=sheet)
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in sheet '{sheet}'. Available: {list(df.columns)}")
        y = df[target_col].values.astype(float)
        X = _build_X_from_smiles_df(df, n_bits=n_bits)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=seed, shuffle=True
    )
    return X_train, X_test, y_train, y_test, df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# ----------------------------
# Baseline core
# ----------------------------
def jaccard_similarity_bool(A, B):
    """
    Compute Jaccard similarity between dense {0,1} arrays (A: n_test x d, B: n_train x d).
    Returns S: n_test x n_train similarity matrix.
    """
    A_bool = (A > 0).astype(bool)
    B_bool = (B > 0).astype(bool)
    D = pairwise_distances(A_bool, B_bool, metric='jaccard', n_jobs=1)
    S = 1.0 - D
    return S

def run_baseline_simple():
    # Fixed defaults
    excel_path = "Nature_SMILES.xlsx"
    sheet = "Reactions"
    target_col = "DDG"
    out_prefix = "baseline_rarf"
    radius = 0.4
    k_cap = None           # set to an int (e.g., 8) if you want a top-K cap per target
    n_estimators = 200
    seed = 42
    test_size = 0.2
    n_bits_per_part = 2048

    t0 = time.time()
    print("\n" + "="*80 + "\n🔷 STEP 1: LOADING EXCEL & BUILDING FINGERPRINTS\n" + "="*80)
    X_train, X_test, y_train, y_test, df_train, df_test = load_features(
        excel_path, sheet=sheet, target_col=target_col, n_bits=n_bits_per_part, test_size=test_size, seed=seed
    )
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

    print("\n" + "="*80 + "\n🔷 STEP 2: NEIGHBORHOODS (JACCARD)\n" + "="*80)
    S = jaccard_similarity_bool(X_test, X_train)  # similarity matrix
    tau = 1.0 - radius
    print(f"   Using radius={radius} → similarity threshold tau={tau:.2f}")

    print("\n" + "="*80 + "\n🔷 STEP 3: PREDICTION LOOP (independent per-target)\n" + "="*80)
    n_test = X_test.shape[0]
    y_pred = np.full(n_test, np.nan)
    n_used = np.zeros(n_test, dtype=int)
    neighbors_per_target = []

    rf_params = dict(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    for i in range(n_test):
        # in-radius neighbors for this target
        nbr_idx = np.where(S[i] >= tau)[0]
        if nbr_idx.size == 0:
            neighbors_per_target.append([])
            continue

        # sort by similarity, pick top-K if requested
        order = np.argsort(S[i, nbr_idx])[::-1]
        nbr_idx = nbr_idx[order]
        if (k_cap is not None) and (nbr_idx.size > k_cap):
            nbr_idx = nbr_idx[:k_cap]

        neighbors_per_target.append(nbr_idx.tolist())
        n_used[i] = nbr_idx.size

        # fit local RF and predict
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train[nbr_idx], y_train[nbr_idx])
        y_pred[i] = rf.predict(X_test[i:i+1])[0]

    print(f"   Finished predictions. Any NaNs: {np.isnan(y_pred).sum()}")

    print("\n" + "="*80 + "\n🔷 STEP 4: METRICS & COST ACCOUNTING\n" + "="*80)
    mask = ~np.isnan(y_pred)
    mae = mean_absolute_error(y_test[mask], y_pred[mask]) if mask.any() else np.nan
    rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])) if mask.any() else np.nan
    r2 = r2_score(y_test[mask], y_pred[mask]) if mask.any() else np.nan

    per_target_cost = int(sum(len(lst) for lst in neighbors_per_target))
    unique_experiment_cost = len(set().union(*[set(lst) for lst in neighbors_per_target]) if neighbors_per_target else set())

    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R^2:  {r2:.4f}")
    print(f"   Per-target cost (naïve sum): {per_target_cost}")
    print(f"   Unique-experiment cost (union): {unique_experiment_cost}")

    # ----------------------------
    # Outputs
    # ----------------------------
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # predictions CSV
    pred_df = df_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df["abs_err"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
    pred_df["neighbors_used_idx"] = [",".join(map(str, lst)) for lst in neighbors_per_target]
    pred_df["n_neighbors"] = n_used
    pred_csv = f"{out_prefix}_baseline_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"   → Wrote {pred_csv}")

    # neighbors JSON
    neigh_json = f"{out_prefix}_baseline_neighbors.json"
    with open(neigh_json, "w") as f:
        json.dump({"neighbors_per_target": neighbors_per_target}, f, indent=2)
    print(f"   → Wrote {neigh_json}")

    # summary JSON
    summary = dict(
        radius=radius, tau=tau, k_cap=k_cap,
        n_estimators=n_estimators, seed=seed, test_size=test_size,
        n_test=int(n_test),
        mae=float(mae), rmse=float(rmse), r2=float(r2),
        per_target_cost=int(per_target_cost),
        unique_experiment_cost=int(unique_experiment_cost),
        runtime_sec=round(time.time() - t0, 3),
    )
    summ_json = f"{out_prefix}_baseline_summary.json"
    with open(summ_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   → Wrote {summ_json}")

    print("\n" + "="*80 + "\n🔷 DONE\n" + "="*80)
    return summary


if __name__ == "__main__":
    run_baseline_simple()
