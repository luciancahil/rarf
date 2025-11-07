#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent RaRF baseline (aligned with prior pipeline) + explicit usage audit.
- Fingerprints via utils.build_X_from_excel
- radius = 0.4 (tau = 0.6), Jaccard on concatenated ECFP4 bits
- TOP_K = 5 per target (to mirror enrich_min_per_target=5)
- Outputs:
    baseline_like_shared_predictions.csv
    baseline_like_shared_coverage.csv
    baseline_like_shared_summary.json
    baseline_usage_edges.csv   (target_idx, train_idx, rank_in_target)
    baseline_usage_matrix.csv  (0/1 indicator matrix: target x train)
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
TOP_K = 5                     # cap neighbors per target
N_EST = 200
SEED = 42
TEST_SIZE = 0.2

OUT_PRED = "baseline_like_shared_predictions.csv"
OUT_COV  = "baseline_like_shared_coverage.csv"
OUT_SUM  = "baseline_like_shared_summary.json"
OUT_EDGES = "baseline_usage_edges.csv"
OUT_MATRIX = "baseline_usage_matrix.csv"


def load_Xy():
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
    print("🔷 BASELINE: Independent RaRF (with explicit usage edges & matrix)")
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
    n_train = X_tr.shape[0]
    y_pred = np.full(n_test, np.nan)
    neigh_lists = []
    neigh_counts = []

    # For usage auditing
    usage_edges = []  # rows: (target_idx, train_idx, rank_in_target)
    usage_matrix = np.zeros((n_test, n_train), dtype=np.uint8)

    for i in range(n_test):
        inrad = np.where(S[i] >= TAU)[0]
        if inrad.size == 0:
            neigh_lists.append([])
            neigh_counts.append(0)
            continue
        # sort by similarity descending
        order = np.argsort(S[i, inrad])[::-1]
        inrad_sorted = inrad[order]
        use_idx = inrad_sorted[:TOP_K] if len(inrad_sorted) > TOP_K else inrad_sorted

        neigh_lists.append(use_idx.tolist())
        neigh_counts.append(len(use_idx))

        # record usage edges & matrix
        for r, tr_idx in enumerate(use_idx):
            usage_edges.append((i, int(tr_idx), r))  # rank within this target
            usage_matrix[i, tr_idx] = 1

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
    unique_cost = int(usage_matrix.sum(axis=0).astype(bool).sum())  # count of train columns used by any target

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

    # coverage summary
    cov_df = pd.DataFrame({
        "target_idx": np.arange(n_test, dtype=int),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    })
    cov_df.to_csv(OUT_COV, index=False)
    print("    →", OUT_COV)

    # usage edges
    edges_df = pd.DataFrame(usage_edges, columns=["target_idx", "train_idx", "rank_in_target"])
    edges_df.to_csv(OUT_EDGES, index=False)
    print("    →", OUT_EDGES)

    # usage matrix (warning: wide file if many train points)
    # include column names as train indices
    matrix_cols = [f"train_{j}" for j in range(n_train)]
    mat_df = pd.DataFrame(usage_matrix, columns=matrix_cols)
    mat_df.insert(0, "target_idx", np.arange(n_test, dtype=int))
    mat_df.to_csv(OUT_MATRIX, index=False)
    print("    →", OUT_MATRIX)

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
        outputs=[OUT_PRED, OUT_COV, OUT_SUM, OUT_EDGES, OUT_MATRIX],
    )
    with open(OUT_SUM, "w") as f:
        json.dump(summary, f, indent=2)
    print("    →", OUT_SUM)

    print("\nDONE.")

if __name__ == "__main__":
    main()
