#!/usr/bin/env python3
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from utils import build_X_from_excel

EXCEL_PATH = "Nature_SMILES.xlsx"
REACTIONS_SHEET = "auto"    # autodetects 'df' in your file
TARGET_COL = "DDG"
N_BITS_PER_PART = 2048
TAU = 0.40
K_MIN, K_MAX = 3, 10
RANDOM_STATE = 13
N_EST = 200
MAX_DEPTH = 10

def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return 1.0 - (inter / uni if uni>0 else 0.0)

def compute_distances(X_te, X_tr):
    D = np.zeros((len(X_te), len(X_tr)), dtype=float)
    for i, xt in enumerate(X_te):
        for j, xr in enumerate(X_tr):
            inter = np.logical_and(xt, xr).sum()
            uni = np.logical_or(xt, xr).sum()
            D[i, j] = 1.0 - (inter / uni if uni>0 else 0.0)
    return D

def get_inradius_sorted(D, tau):
    inrad_lists = []
    for i in range(D.shape[0]):
        inrad = np.where(D[i] < tau)[0].tolist()
        inrad_sorted = sorted(inrad, key=lambda j: D[i, j])
        inrad_lists.append(inrad_sorted)
    return inrad_lists

def baseline_k_predict(X_tr, y_tr, X_te, y_te, inrad_lists, k):
    usage = np.zeros((len(X_te), len(X_tr)), dtype=int)
    preds = []
    neigh_counts = []
    for i in range(len(X_te)):
        chosen = inrad_lists[i][:k]   # simply take the k nearest in-radius
        neigh_counts.append(len(chosen))
        if len(chosen)==0:
            preds.append(np.nan)
        else:
            rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
            rf.fit(X_tr[chosen], y_tr[chosen])
            preds.append(float(rf.predict(X_te[i].reshape(1,-1))[0]))
        for j in chosen:
            usage[i, j] = 1
    return np.array(preds), usage, neigh_counts

def shared_k_predict(X_tr, y_tr, X_te, y_te, inrad_lists, k):
    # Greedy shared selection until every target has >=k or no utility remains
    coverage = np.zeros(len(X_te), dtype=int)
    selected = set()
    while True:
        utility = np.zeros(len(X_tr), dtype=int)
        need_more = False
        for i in range(len(X_te)):
            if coverage[i] < k:
                need_more = True
                for j in inrad_lists[i]:
                    if j not in selected:
                        utility[j] += 1
        if not need_more:
            break
        if utility.max() == 0:
            break
        j_star = int(np.argmax(utility))
        selected.add(j_star)
        for i in range(len(X_te)):
            if coverage[i] < k and j_star in inrad_lists[i]:
                coverage[i] += 1
    selected = sorted(selected)

    usage = np.zeros((len(X_te), len(X_tr)), dtype=int)
    preds = []
    neigh_counts = []
    for i in range(len(X_te)):
        chosen = [j for j in inrad_lists[i] if j in selected][:k]
        neigh_counts.append(len(chosen))
        if len(chosen)==0:
            preds.append(np.nan)
        else:
            rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
            rf.fit(X_tr[chosen], y_tr[chosen])
            preds.append(float(rf.predict(X_te[i].reshape(1,-1))[0]))
        for j in chosen:
            usage[i, j] = 1
    return np.array(preds), usage, neigh_counts

def metrics(y_true, y_pred, usage, neigh_counts, k):
    mask = ~np.isnan(y_pred)
    if mask.any():
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        rmse = mean_squared_error(y_true[mask], y_pred[mask])**0.5
        yt = y_true[mask]; yp = y_pred[mask]
        ss_res = float(((yt-yp)**2).sum()); ss_tot = float(((yt-yt.mean())**2).sum())
        r2 = 1 - ss_res/ss_tot if ss_tot>0 else float("nan")
    else:
        mae = rmse = r2 = float("nan")
    union = int((usage.sum(axis=0) > 0).sum())
    pct_ge_k = float((np.array(neigh_counts) >= k).mean() * 100.0)
    mean_neighbors = float(np.mean(neigh_counts))
    return dict(MAE=mae, RMSE=rmse, R2=r2, UNION=union, PCT_GE_K=pct_ge_k, MEAN_NEI=mean_neighbors)

def maybe_parity(fname, y_true, y_pred):
    mask = ~np.isnan(y_pred)
    if not mask.any():
        return
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(y_true[mask], y_pred[mask], s=14)
    lims = [min(y_true[mask].min(), y_pred[mask].min()), max(y_true[mask].max(), y_pred[mask].max())]
    plt.plot(lims, lims)
    plt.xlabel("Measured ΔΔG‡ (kcal/mol)")
    plt.ylabel("Predicted ΔΔG‡")
    plt.title(fname.replace(".png",""))
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()

def main():
    # Data + fixed split
    X, y = build_X_from_excel(EXCEL_PATH, reactions_sheet=REACTIONS_SHEET, target_col=TARGET_COL, nbits=N_BITS_PER_PART)
    X = np.asarray(X); y = np.asarray(y, dtype=float)
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.20, random_state=RANDOM_STATE, shuffle=True)

    # Distance + in-radius
    D = compute_distances(X_te, X_tr)
    inrad = get_inradius_sorted(D, TAU)

    rows = []
    for k in range(K_MIN, K_MAX+1):
        # Baseline-k
        ypb, U_b, nb = baseline_k_predict(X_tr, y_tr, X_te, y_te, inrad, k)
        mb = metrics(y_te, ypb, U_b, nb, k)
        mb.update(dict(METHOD="Baseline-k", K=k, TAU=TAU))
        rows.append(mb)
        if k in (K_MIN, K_MAX):
            maybe_parity(f"parity_baseline_k{k}.png", y_te, ypb)

        # Shared-k
        yps, U_s, ns = shared_k_predict(X_tr, y_tr, X_te, y_te, inrad, k)
        ms = metrics(y_te, yps, U_s, ns, k)
        ms.update(dict(METHOD="Shared(k)", K=k, TAU=TAU))
        rows.append(ms)
        if k in (K_MIN, K_MAX):
            maybe_parity(f"parity_shared_k{k}.png", y_te, yps)

    df = pd.DataFrame(rows)
    df.to_csv("sweep_results.csv", index=False)
    print(df)

    # Plot MAE vs UNION
    plt.figure()
    for method, g in df.groupby("METHOD"):
        g_sorted = g.sort_values("UNION")
        plt.plot(g_sorted["UNION"], g_sorted["MAE"], marker="o", label=method)
    plt.xlabel("Union size (unique training experiments)")
    plt.ylabel("MAE (kcal/mol)")
    plt.title(f"Efficiency–Accuracy curve (τ={TAU}, k={K_MIN}..{K_MAX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mae_vs_union.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
