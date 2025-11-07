#!/usr/bin/env python3
# RaRF baseline with neighbor introspection
import json, time, argparse, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import build_X_from_excel, get_distances

DEFAULTS = dict(
    excel="Nature_SMILES.xlsx",
    sheet=None,
    target="DDG",
    bits=2048,
    radius=0.45,
    fallback=0.60,
    trees=300,
    test_size=0.20,
    seed=42,
)

def parse_args():
    p = argparse.ArgumentParser(description="Independent RaRF baseline using utils.get_distances (with neighbor logging)")
    p.add_argument("--excel", default=DEFAULTS["excel"], help="Path to workbook (xlsx)")
    p.add_argument("--sheet", default=DEFAULTS["sheet"], help="Reactions sheet name (default: auto-detect)")
    p.add_argument("--target", default=DEFAULTS["target"], help="Target column (default: DDG)")
    p.add_argument("--bits", type=int, default=DEFAULTS["bits"], help="Bits per part for Morgan FP (default: 2048)")
    p.add_argument("--radius", type=float, default=DEFAULTS["radius"], help="Distance cutoff D<=R (default: 0.45)")
    p.add_argument("--fallback", type=float, default=DEFAULTS["fallback"], help="Fallback radius if empty (default: 0.60)")
    p.add_argument("--trees", type=int, default=DEFAULTS["trees"], help="RandomForest trees (default: 300)")
    p.add_argument("--test_size", type=float, default=DEFAULTS["test_size"], help="Test fraction (default: 0.20)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed (default: 42)")
    return p.parse_args()

def main():
    args = parse_args()
    t0 = time.time()
    print("="*80)
    print("🔷 BASELINE (user utils): Independent RaRF using utils.get_distances")
    print("="*80)

    print("\n[1] Loading & fingerprinting ...")
    X, y = build_X_from_excel(args.excel, reactions_sheet=args.sheet, target_col=args.target, nbits=args.bits)
    print(f"    X: {X.shape} | y: {y.shape} | bits/part={args.bits} | sheet={args.sheet or 'auto'}")

    print("\n[2] Train/test split (80/20, seed={}) ...".format(args.seed))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, shuffle=True)

    print("\n[3] Distances (test vs train) via utils.get_distances ...")
    D = get_distances(X_tr, X_te)
    if D.shape == (X_tr.shape[0], X_te.shape[0]):
        D = D.T
    S = 1.0 - D
    tau_primary = 1.0 - args.radius
    tau_fallback = 1.0 - args.fallback
    print(f"    radius={args.radius:.2f} → tau={tau_primary:.2f}")

    print("\n[4] Per-target RaRF (all in radius) ...")
    rf_params = dict(n_estimators=args.trees, random_state=args.seed, n_jobs=-1)
    n_test, n_train = X_te.shape[0], X_tr.shape[0]
    y_pred = np.full(n_test, np.nan)

    # logging
    neigh_counts = []
    neigh_idx_list = []
    selection_method = []  # 'primary' | 'fallback' | 'nearest'
    usage_edges = []       # rows: (target_idx, train_idx, rank, method)
    usage_matrix = np.zeros((n_test, n_train), dtype=np.uint8)

    for i in range(n_test):
        # try primary radius
        inrad = np.where(S[i] >= tau_primary)[0]
        method = "primary"
        if inrad.size == 0:
            # try fallback radius
            inrad = np.where(S[i] >= tau_fallback)[0]
            method = "fallback"
        if inrad.size == 0:
            # nearest neighbor fallback
            inrad = np.array([int(np.argmax(S[i]))], dtype=int)
            method = "nearest"

        order = np.argsort(S[i, inrad])[::-1]
        use_idx = inrad[order]

        # log
        neigh_counts.append(int(len(use_idx)))
        neigh_idx_list.append(use_idx.tolist())
        selection_method.append(method)
        for r, tr_idx in enumerate(use_idx):
            usage_edges.append((i, int(tr_idx), r, method))
            usage_matrix[i, tr_idx] = 1

        # fit & predict
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr[use_idx], y_tr[use_idx])
        y_pred[i] = rf.predict(X_te[i:i+1])[0]

    print("\n[5] Metrics & neighbor coverage ...")
    mask = ~np.isnan(y_pred)
    mae  = float(mean_absolute_error(y_te[mask], y_pred[mask])) if mask.any() else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_te[mask], y_pred[mask]))) if mask.any() else float("nan")
    r2   = float(r2_score(y_te[mask], y_pred[mask])) if mask.any() else float("nan")
    per_target_cost = int(sum(neigh_counts))                   # total neighbors across all targets (no dedup)
    unique_experiment_cost = int(usage_matrix.sum(axis=0).astype(bool).sum())  # unique train points used

    # also count how many targets used each method
    method_counts = {m: selection_method.count(m) for m in ("primary", "fallback", "nearest")}

    print(f"    MAE : {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R^2 : {r2:.4f}")
    print(f"    Per-target cost (no-dedup): {per_target_cost}")
    print(f"    Unique neighbors used (deduped union): {unique_experiment_cost}")
    print(f"    Selection methods → primary: {method_counts['primary']}, fallback: {method_counts['fallback']}, nearest: {method_counts['nearest']}")

    # [6] Save artifacts
    # preds.csv: include neighbors and method
    preds_df = pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred,
        "abs_err": np.abs(y_te - y_pred),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_idx_list],
        "method": selection_method,
    })
    preds_df.to_csv("preds.csv", index=False); print("    → preds.csv")

    # coverage report per test
    cov_df = pd.DataFrame({
        "target_idx": np.arange(n_test, dtype=int),
        "n_neighbors": neigh_counts,
        "neighbors_idx": [",".join(map(str, lst)) for lst in neigh_idx_list],
        "method": selection_method,
    })
    cov_df.to_csv("rarf_coverage_report.csv", index=False); print("    → rarf_coverage_report.csv")

    # usage edges and matrix
    pd.DataFrame(usage_edges, columns=["target_idx", "train_idx", "rank", "method"]).to_csv("usage_edges.csv", index=False); print("    → usage_edges.csv")

    matrix_cols = [f"train_{j}" for j in range(n_train)]
    mat_df = pd.DataFrame(usage_matrix, columns=matrix_cols)
    mat_df.insert(0, "target_idx", np.arange(n_test, dtype=int))
    mat_df.to_csv("usage_matrix.csv", index=False); print("    → usage_matrix.csv")

    # summary
    with open("summary.json", "w") as f:
        json.dump(dict(
            excel=args.excel, sheet=args.sheet or "auto", target_col=args.target,
            radius=args.radius, tau_primary=tau_primary, fallback=args.fallback, tau_fallback=tau_fallback,
            bits_per_part=args.bits,
            rf_estimators=args.trees, seed=args.seed, test_size=args.test_size,
            n_test=int(n_test), n_train=int(n_train),
            mae=mae, rmse=rmse, r2=r2,
            per_target_cost=per_target_cost,
            unique_experiment_cost=unique_experiment_cost,
            method_counts=method_counts,
            runtime_sec=round(time.time() - t0, 3),
            outputs=["preds.csv", "rarf_coverage_report.csv", "usage_edges.csv", "usage_matrix.csv", "summary.json"],
        ), f, indent=2); print("    → summary.json")

    print("\nDONE.")

if __name__ == "__main__":
    main()
