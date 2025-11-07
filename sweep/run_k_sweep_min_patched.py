#!/usr/bin/env python3
"""
Patched k-sweep runner (fast distances + CLI flags)

Based on the user's original run_k_sweep_min.py, but:
- Uses utils.get_distances (vectorized) instead of nested loops (much faster)
- Adds CLI flags: --tau, --kmin, --kmax, --seed, --trees, --depth, --test_size
- Treats REACTIONS_SHEET="auto" as auto-detect (i.e., passes None to build_X_from_excel)
- Optionally saves parity plots only for endpoints (kmin, kmax)

Outputs:
- sweep_results.csv
- mae_vs_union.png
- parity_baseline_k{K}.png (for K=kmin,kmax)
- parity_shared_k{K}.png (for K=kmin,kmax)
"""
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from utils import build_X_from_excel, get_distances

DEFAULTS = dict(
    excel="Nature_SMILES.xlsx",
    sheet="auto",            # auto-detect 'df' in your file
    target="DDG",
    bits=2048,
    tau=0.40,
    kmin=3,
    kmax=10,
    seed=13,
    trees=200,
    depth=10,
    test_size=0.20,
)

def parse_args():
    p = argparse.ArgumentParser(description="Sweep k for Baseline-k vs Shared(k) at fixed τ")
    p.add_argument("--excel", default=DEFAULTS["excel"])
    p.add_argument("--sheet", default=DEFAULTS["sheet"], help='"auto" to autodetect, or provide a sheet name')
    p.add_argument("--target", default=DEFAULTS["target"])
    p.add_argument("--bits", type=int, default=DEFAULTS["bits"])
    p.add_argument("--tau", type=float, default=DEFAULTS["tau"], help="Distance threshold; neighbors have D < τ")
    p.add_argument("--kmin", type=int, default=DEFAULTS["kmin"])
    p.add_argument("--kmax", type=int, default=DEFAULTS["kmax"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--trees", type=int, default=DEFAULTS["trees"])
    p.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    p.add_argument("--test_size", type=float, default=DEFAULTS["test_size"])
    return p.parse_args()

def get_inradius_sorted(D, tau):
    inrad_lists = []
    for i in range(D.shape[0]):
        inrad = np.where(D[i] < tau)[0].tolist()
        inrad_sorted = sorted(inrad, key=lambda j: D[i, j])
        inrad_lists.append(inrad_sorted)
    return inrad_lists

def baseline_k_predict(X_tr, y_tr, X_te, inrad_lists, k, trees, depth, seed):
    usage = np.zeros((len(X_te), len(X_tr)), dtype=int)
    preds = []
    neigh_counts = []
    for i in range(len(X_te)):
        chosen = inrad_lists[i][:k]   # simply take the k nearest in-radius
        neigh_counts.append(len(chosen))
        if len(chosen)==0:
            preds.append(np.nan)
        else:
            rf = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=-1)
            rf.fit(X_tr[chosen], y_tr[chosen])
            preds.append(float(rf.predict(X_te[i].reshape(1,-1))[0]))
        for j in chosen:
            usage[i, j] = 1
    return np.array(preds), usage, neigh_counts

def shared_k_predict(X_tr, y_tr, X_te, inrad_lists, k, trees, depth, seed):
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
            rf = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=-1)
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
    args = parse_args()

    # Reactions sheet handling
    sheet = None if (args.sheet is None or args.sheet.lower()=="auto") else args.sheet

    # Data + split
    X, y = build_X_from_excel(args.excel, reactions_sheet=sheet, target_col=args.target, nbits=args.bits)
    X = np.asarray(X); y = np.asarray(y, dtype=float)
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=args.test_size, random_state=args.seed, shuffle=True)

    # Distance + in-radius (fast)
    D = get_distances(X_tr, X_te).T   # utils returns (n_test x n_train) if (X_train,X_test); we passed (X_tr,X_te)
    inrad = get_inradius_sorted(D, args.tau)

    rows = []
    for k in range(args.kmin, args.kmax+1):
        # Baseline-k
        ypb, U_b, nb = baseline_k_predict(X_tr, y_tr, X_te, inrad, k, args.trees, args.depth, args.seed)
        mb = metrics(y_te, ypb, U_b, nb, k)
        mb.update(dict(METHOD="Baseline-k", K=k, TAU=args.tau))
        rows.append(mb)
        if k in (args.kmin, args.kmax):
            maybe_parity(f"parity_baseline_k{k}.png", y_te, ypb)

        # Shared-k
        yps, U_s, ns = shared_k_predict(X_tr, y_tr, X_te, inrad, k, args.trees, args.depth, args.seed)
        ms = metrics(y_te, yps, U_s, ns, k)
        ms.update(dict(METHOD="Shared(k)", K=k, TAU=args.tau))
        rows.append(ms)
        if k in (args.kmin, args.kmax):
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
    plt.title(f"Efficiency–Accuracy curve (τ={args.tau}, k={args.kmin}..{args.kmax})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mae_vs_union.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
