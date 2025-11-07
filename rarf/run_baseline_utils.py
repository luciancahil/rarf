#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent RaRF baseline (utils-based, auto-detect sheet) + explicit usage audit.
- Fingerprints via utils.build_X_from_excel (fault-tolerant utils)
- radius = 0.4 (=> tau = 0.6)
- TOP_K = None  → use ALL in-radius neighbors (true RaRF)
         = int → cap neighbors per target to this K (for apples-to-apples w/ shared)
"""
import json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import build_X_from_excel, jaccard_similarity_bool

EXCEL_PATH = "Nature_SMILES.xlsx"
REACTIONS_SHEET = None
TARGET_COL = "DDG"
N_BITS_PER_PART = 2048
RADIUS = 0.4
TAU = 1.0 - RADIUS
TOP_K = None
N_EST = 200
SEED = 42
TEST_SIZE = 0.2

OUT_PRED = "baseline_like_shared_predictions.csv"
OUT_COV  = "baseline_like_shared_coverage.csv"
OUT_SUM  = "baseline_like_shared_summary.json"
OUT_EDGES = "baseline_usage_edges.csv"
OUT_MATRIX = "baseline_usage_matrix.csv"

def main():
    t0 = time.time()
    print("="*80)
    print("🔷 BASELINE (utils): Independent RaRF with explicit usage audit")
    print("="*80)

    print("\n[1] Loading & fingerprinting via utils.build_X_from_excel ...")
    X, y = build_X_from_excel(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    print(f"    X shape: {X.shape} | y: {y.shape} | bits/part={N_BITS_PER_PART} | sheet={REACTIONS_SHEET or 'auto'}")

    print("\n[2] Train/test split (80/20, seed=42) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    print(f"    X_tr: {X_tr.shape} | X_te: {X_te.shape}")

    print("\n[3] Jaccard similarity (test vs train) via utils.jaccard_similarity_bool ...")
    S = jacc_similarity = jaccard_similarity_bool(X_te, X_tr)
    print(f"    radius={RADIUS} → tau={TAU:.2f} | TOP_K={'ALL' if TOP_K is None else TOP_K}")

    print("\n[4] Per-target neighbor selection & prediction ...")
    rf_params = dict(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    n_test, n_train = X_te.shape[0], X_tr.shape[0]
    y_pred = np.full(n_test, np.nan)
    neigh_lists, neigh_counts = [], []

    usage_edges = []
    usage_matrix = np.zeros((n_test, n_train), dtype=np.uint8)

    for i in range(n_test):
        inrad = np.where(S[i] >= TAU)[0]
        if inrad.size == 0:
            neigh_lists.append([])
            neigh_counts.append(0)
            continue
        order = np.argsort(S[i, inrad])[::-1]
        inrad_sorted = inrad[order]
        use_idx = inrad_sorted if TOP_K is None else inrad_sorted[:TOP_K]

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

    per_target_cost = int(sum(len(lst) for lst in neigh_lists))
    unique_cost     = int(usage_matrix.sum(axis=0).astype(bool).sum())

    print(f"    MAE : {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R^2 : {r2:.4f}")
    print(f"    Per-target cost (no-dedup): {per_target_cost}")
    print(f"    Unique-experiment cost (deduped union): {unique_cost}")

    print("\n[6] Writing outputs ...")
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred,
        "abs_err": np.abs(y_te - y_pred),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    }).to_csv(OUT_PRED, index=False); print("    →", OUT_PRED)

    pd.DataFrame({
        "target_idx": np.arange(n_test, dtype=int),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_lists],
    }).to_csv(OUT_COV, index=False); print("    →", OUT_COV)

    pd.DataFrame(usage_edges, columns=["target_idx", "train_idx", "rank_in_target"]).to_csv(OUT_EDGES, index=False); print("    →", OUT_EDGES)

    matrix_cols = [f"train_{j}" for j in range(n_train)]
    mat_df = pd.DataFrame(usage_matrix, columns=matrix_cols)
    mat_df.insert(0, "target_idx", np.arange(n_test, dtype=int))
    mat_df.to_csv(OUT_MATRIX, index=False); print("    →", OUT_MATRIX)

    with open(OUT_SUM, "w") as f:
        json.dump(dict(
            excel=EXCEL_PATH, sheet=REACTIONS_SHEET, target_col=TARGET_COL,
            radius=RADIUS, tau=TAU, top_k=TOP_K,
            bits_per_part=N_BITS_PER_PART,
            rf_estimators=N_EST, seed=SEED, test_size=TEST_SIZE,
            n_test=int(n_test),
            mae=mae, rmse=rmse, r2=r2,
            per_target_cost=per_target_cost,
            unique_experiment_cost=unique_cost,
            runtime_sec=round(time.time() - t0, 3),
            outputs=[OUT_PRED, OUT_COV, OUT_SUM, OUT_EDGES, OUT_MATRIX],
        ), f, indent=2); print("    →", OUT_SUM)

    print("\nDONE.")

if __name__ == "__main__":
    main()
