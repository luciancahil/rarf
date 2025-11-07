#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent RaRF baseline (aligned with the earlier RaRF-Shared pipeline).
- Fingerprints via utils.build_X_from_excel (same as shared)
- radius = 0.4 (tau = 0.6) with Jaccard on concatenated ECFP4 bits
- Top-K cap per target = 5 (to mirror enrich_min_per_target=5 used before)
- No cross-target sharing; each target picks its own neighbors independently
- Outputs:
    baseline_like_shared_predictions.csv
    baseline_like_shared_coverage.csv
    baseline_like_shared_summary.json
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances, mean_absolute_error, mean_squared_error, r2_score

# --- Config (fixed to keep "no-args") ---
EXCEL_PATH = "Nature_SMILES.xlsx"
REACTIONS_SHEET = "Reactions"
TARGET_COL = "DDG"
N_BITS_PER_PART = 2048
RADIUS = 0.4
TAU = 1.0 - RADIUS
TOP_K = 5                     # cap neighbors per target (to match enrich_min_per_target=5)
N_EST = 200
SEED = 42
TEST_SIZE = 0.2

OUT_PRED = "baseline_like_shared_predictions.csv"
OUT_COV  = "baseline_like_shared_coverage.csv"
OUT_SUM  = "baseline_like_shared_summary.json"


def load_Xy():
    # Use the same builder we've been using in prior runs
    from utils import build_X_from_excel
    X, y = build_X_from_excel(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    return X, y

def jaccard_similarity(A, B):
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    D = pairwise_distances(A, B, metric="jaccard", n_jobs=1)
    return 1.0 - D

def main():
    t0 = time.time()
    print("="*80)
    print("🔷 BASELINE: Independent RaRF (aligned with prior pipeline)")
    print("="*80)

    print("\n[1] Loading & fingerprinting via utils.build_X_from_excel ...")
    X, y = load_Xy()
    print(f"    X shape: {X.shape} | y: {y.shape}")

    print("\n[2] Train/test split (80/20, seed=42) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    print(f"    X_tr: {X_tr.shape} | X_te: {X_te.shape}")

    print("\n[3] Jaccard similarity (test vs train) ...")
    S = jaccard_similarity(X_te, X_tr)
    print(f"    radius={RADIUS} → tau={TAU:.2f}")

    print("\n[4] Per-target neighbor selection & prediction (TOP_K=5) ...")
    rf_params = dict(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    n_test = X_te.shape[0]
    y_pred = np.full(n_test, np.nan)
    neigh_lists = []
    neigh_counts = []

    for i in range(n_test):
        inrad = np.where(S[i] >= TAU)[0]
        if inrad.size == 0:
            neigh_lists.append([])
            neigh_counts.append(0)
            continue
        # sort by similarity descending, then cap at TOP_K
        order = np.argsort(S[i, inrad])[::-1]
        inrad_sorted = inrad[order]
        use_idx = inrad_sorted[:TOP_K] if len(inrad_sorted) > TOP_K else inrad_sorted
        neigh_lists.append(use_idx.tolist())
        neigh_counts.append(len(use_idx))
        # fit local RF
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr[use_idx], y_tr[use_idx])
        y_pred[i] = rf.predict(X_te[i:i+1])[0]

    print("    Done. NaN preds:", int(np.isnan(y_pred).sum()))

    print("\n[5] Metrics & cost accounting ...")
    mask = ~np.isnan(y_pred)
    mae  = float(mean_absolute_error(y_te[mask], y_pred[mask])) if mask.any() else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_te[mask], y_pred[mask]))) if mask.any() else float("nan")
    r2   = float(r2_score(y_te[mask], y_pred[mask])) if mask.any() else float("nan")

    per_target_cost = int(sum(len(lst) for lst in neigh_lists))
    unique_cost = len(set().union(*[set(lst) for lst in neigh_lists]) if neigh_lists else set())

    print(f"    MAE : {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R^2 : {r2:.4f}")
    print(f"    Per-target cost (sum of K's): {per_target_cost}")
    print(f"    Unique-experiment cost (deduped union): {unique_cost}")

    print("\n[6] Writing outputs ...")
    # predictions
    pred_df = pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred,
        "abs_err": np.abs(y_te - y_pred),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    })
    pred_df.to_csv(OUT_PRED, index=False)
    print("    →", OUT_PRED)

    # coverage
    cov_df = pd.DataFrame({
        "target_idx": np.arange(n_test, dtype=int),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    })
    cov_df.to_csv(OUT_COV, index=False)
    print("    →", OUT_COV)

    # summary
    summary = dict(
        excel=EXCEL_PATH,
        sheet=REACTIONS_SHEET,
        target_col=TARGET_COL,
        radius=RADIUS,
        tau=TAU,
        top_k=TOP_K,
        n_estimators=N_EST,
        seed=SEED,
        test_size=TEST_SIZE,
        n_test=int(n_test),
        mae=mae, rmse=rmse, r2=r2,
        per_target_cost=per_target_cost,
        unique_experiment_cost=unique_cost,
        runtime_sec=round(time.time() - t0, 3),
    )
    with open(OUT_SUM, "w") as f:
        json.dump(summary, f, indent=2)
    print("    →", OUT_SUM)

    print("\nDONE.")

if __name__ == "__main__":
    main()
