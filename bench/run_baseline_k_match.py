#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline-k (independent RaRF with k-NN control) — self-contained
Relies only on utils.build_X_from_excel.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ---- user knobs ----
EXCEL_PATH = "Nature_SMILES.xlsx"
REACTIONS_SHEET = "Reactions"
TARGET_COL = "DDG"
N_BITS_PER_PART = 2048
TAU = 0.40
K = 5
RANDOM_STATE = 13
N_EST = 200
MAX_DEPTH = 10
# --------------------

from utils import build_X_from_excel

def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 - (inter / union if union>0 else 0.0)

def main():
    out_dir = Path(".")
    X, y = build_X_from_excel(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    X = np.asarray(X); y = np.asarray(y, dtype=float)

    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.20, random_state=RANDOM_STATE, shuffle=True)

    # D(test,train)
    D = np.zeros((len(X_te), len(X_tr)), dtype=float)
    for i, xt in enumerate(X_te):
        for j, xr in enumerate(X_tr):
            D[i, j] = jaccard(xt, xr)

    usage_mat = np.zeros((len(X_te), len(X_tr)), dtype=int)
    usage_edges = []
    preds = []
    for i in range(len(X_te)):
        inrad = np.where(D[i] < TAU)[0].tolist()
        inrad_sorted = sorted(inrad, key=lambda j: D[i, j])
        chosen = inrad_sorted[:K]
        if len(chosen)==0:
            preds.append(np.nan)
        else:
            rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
            rf.fit(X_tr[chosen], y_tr[chosen])
            preds.append(float(rf.predict(X_te[i].reshape(1,-1))[0]))
        for rank, j in enumerate(chosen):
            usage_mat[i, j] = 1
            usage_edges.append((i, j, rank))

    mask = ~np.isnan(preds)
    mae = mean_absolute_error(y_te[mask], np.array(preds)[mask])
    rmse = mean_squared_error(y_te[mask], np.array(preds)[mask]) ** 0.5
    yt = y_te[mask]; yp = np.array(preds)[mask]
    ss_res = float(((yt-yp)**2).sum()); ss_tot = float(((yt-yt.mean())**2).sum())
    r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")

    union = int((usage_mat.sum(axis=0) > 0).sum())

    pd.DataFrame({
        "target_idx": list(range(len(X_te))),
        "y_true": y_te,
        "y_pred": preds,
        "abs_err": np.abs(y_te - np.array(preds))
    }).to_csv(out_dir/"preds_baseline_k.csv", index=False)

    cols = [f"train_{j}" for j in range(len(X_tr))]
    um = pd.DataFrame(usage_mat, columns=cols); um.insert(0, "target_idx", list(range(len(X_te))))
    um.to_csv(out_dir/"usage_matrix_baseline_k.csv", index=False)

    pd.DataFrame(usage_edges, columns=["target_idx","train_idx","rank"]).to_csv(out_dir/"usage_edges_baseline_k.csv", index=False)

    summary = dict(setup="Baseline-k", tau=TAU, k=K, n_test=int(len(X_te)),
                   mae=float(mae), rmse=float(rmse), r2=float(r2), union=int(union),
                   mean_neighbors_per_target=float(usage_mat.sum(axis=1).mean()))
    Path("summary_baseline_k.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
