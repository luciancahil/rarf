#!/usr/bin/env python3
# Baseline-B (round-robin union budget) — AUTO reactions sheet
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from utils import build_X_from_excel

EXCEL_PATH = "Nature_SMILES.xlsx"
REACTIONS_SHEET = "auto"
TARGET_COL = "DDG"
N_BITS_PER_PART = 2048
TAU = 0.40
K = 5               # desired per-target, used only for reporting how many meet >=K
BUDGET = 80         # global union budget
RANDOM_STATE = 13
N_EST = 200
MAX_DEPTH = 10

def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return 1.0 - (inter / uni if uni>0 else 0.0)

def main():
    X, y = build_X_from_excel(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    X = np.asarray(X); y = np.asarray(y, dtype=float)

    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.20, random_state=RANDOM_STATE, shuffle=True)

    # Distances test→train
    D = np.zeros((len(X_te), len(X_tr)), dtype=float)
    for i, xt in enumerate(X_te):
        for j, xr in enumerate(X_tr):
            D[i, j] = jaccard(xt, xr)

    # For each target, sorted in-radius neighbors (closest first)
    inrad_lists = []
    for i in range(len(X_te)):
        inrad = np.where(D[i] < TAU)[0].tolist()
        inrad_sorted = sorted(inrad, key=lambda j: D[i, j])
        inrad_lists.append(inrad_sorted)

    # Round-robin union fill up to BUDGET
    selected = set()
    cursors = [0]*len(X_te)  # next candidate index per target
    progressed = True
    while len(selected) < BUDGET and progressed:
        progressed = False
        for i in range(len(X_te)):
            lst = inrad_lists[i]
            # advance cursor until we find an as-yet-unselected neighbor
            while cursors[i] < len(lst) and lst[cursors[i]] in selected:
                cursors[i] += 1
            if cursors[i] < len(lst) and len(selected) < BUDGET:
                selected.add(lst[cursors[i]])
                cursors[i] += 1
                progressed = True
            if len(selected) >= BUDGET:
                break
    selected = sorted(selected)

    # Train per target from selected subset (up to K nearest selected)
    usage_mat = np.zeros((len(X_te), len(X_tr)), dtype=int)
    usage_edges = []
    preds = []
    neigh_counts = []
    for i in range(len(X_te)):
        chosen = [j for j in inrad_lists[i] if j in selected][:K]
        neigh_counts.append(len(chosen))
        if len(chosen)==0:
            preds.append(np.nan)
        else:
            rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
            rf.fit(X_tr[chosen], y_tr[chosen])
            preds.append(float(rf.predict(X_te[i].reshape(1,-1))[0]))
        for rank, j in enumerate(chosen):
            usage_mat[i, j] = 1
            usage_edges.append((i, j, rank))

    # Metrics
    mask = ~np.isnan(preds)
    mae = mean_absolute_error(y_te[mask], np.array(preds)[mask]) if mask.any() else float('nan')
    rmse = (mean_squared_error(y_te[mask], np.array(preds)[mask]) ** 0.5) if mask.any() else float('nan')
    yt = y_te[mask]; yp = np.array(preds)[mask]
    if mask.any() and yt.size>1:
        ss_res = float(((yt-yp)**2).sum()); ss_tot = float(((yt-yt.mean())**2).sum())
        r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")
    else:
        r2 = float("nan")
    union = int((usage_mat.sum(axis=0) > 0).sum())
    pct_ge_k = float((np.array(neigh_counts) >= K).mean() * 100.0)

    # Outputs
    pd.DataFrame({
        "target_idx": list(range(len(X_te))),
        "y_true": y_te,
        "y_pred": preds,
        "abs_err": np.abs(y_te - np.array(preds)),
        "n_selected_neighbors": neigh_counts
    }).to_csv("preds_baseline_B.csv", index=False)

    cols = [f"train_{j}" for j in range(len(X_tr))]
    um = pd.DataFrame(usage_mat, columns=cols); um.insert(0, "target_idx", list(range(len(X_te))))
    um.to_csv("usage_matrix_baseline_B.csv", index=False)

    pd.DataFrame(usage_edges, columns=["target_idx","train_idx","rank"]).to_csv("usage_edges_baseline_B.csv", index=False)

    summary = dict(setup="Baseline-B", tau=TAU, k=K, budget=BUDGET, n_test=int(len(X_te)),
                   mae=float(mae), rmse=float(rmse), r2=float(r2), union=int(union),
                   pct_targets_ge_k=float(pct_ge_k))
    Path("summary_baseline_B.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
