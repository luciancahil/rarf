
"""
RaRF Baseline (per-target local RF; no shared selection)

- Loads Nature_SMILES.xlsx (same schema you've been using)
- Builds Morgan fingerprints (nuc, lig, imine, solvent) and concatenates
- For each test target:
    neighbors = {train j | Jaccard(test_i, train_j) >= tau}, where tau = 1 - radius
    (optional) keep only top-k most similar neighbors (k_cap)
    fit RandomForest on those neighbors and predict for the target
- Writes: rarf_baseline_predictions.csv
- Plots: parity_baseline.png

Adjust params in the CONFIG block below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances

from utils import smi2morgan

# =====================
# CONFIG
# =====================
XLS_PATH = "Nature_SMILES.xlsx"
TRAIN_FRAC = 0.80
RADIUS = 0.4            # tau = 1 - RADIUS
K_CAP = None            # e.g., 10 or 20 to limit to top-k neighbors; None = use all in-radius
N_ESTIMATORS = 200
SEED = 42
OUT_PREFIX = "baseline"

# =====================
# Helpers
# =====================
def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def map_smiles_column(df_main, df_lookup, key_main, key_lookup, smiles_col):
    m = df_lookup.set_index(key_lookup)[smiles_col].to_dict()
    return df_main[key_main].map(m).astype(str).values

def build_X_from_excel(xls_path, nbits=2048):
    xls = pd.ExcelFile(xls_path)
    df = pd.read_excel(xls, sheet_name="df")
    df_im = pd.read_excel(xls, sheet_name="imine")
    df_li = pd.read_excel(xls, sheet_name="ligand")
    df_so = pd.read_excel(xls, sheet_name="solvent")
    df_nu = pd.read_excel(xls, sheet_name="nuc")

    smiles_i = map_smiles_column(df, df_im, "Imine", "Imine", "SMILES_i")
    smiles_l = map_smiles_column(df, df_li, "Ligand", "ligand", "SMILES_l")
    smiles_s = map_smiles_column(df, df_so, "Solvent", "solvent", "SMILES_s")
    smiles_n = map_smiles_column(df, df_nu, "Nucleophile", "Nucleophile", "SMILES_n")

    nu = np.vstack([smi2morgan(s, nbits=nbits) for s in smiles_n])
    li = np.vstack([smi2morgan(s, nbits=nbits) for s in smiles_l])
    im = np.vstack([smi2morgan(s, nbits=nbits) for s in smiles_i])
    so = np.vstack([smi2morgan(s, nbits=nbits) for s in smiles_s])

    X = np.hstack([nu, li, im, so]).astype(np.uint8)
    y = df["DDG"].values
    meta = df.copy()
    return X, y, meta

# =====================
# Main
# =====================
if __name__ == "__main__":
    print("\n=== STEP 1: Load + featurize ===")
    X, y, meta = build_X_from_excel(XLS_PATH, nbits=2048)
    print(f"X: {X.shape}, y: {y.shape}")

    print("\n=== STEP 2: Train/test split ===")
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, train_size=TRAIN_FRAC, random_state=SEED
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n=== STEP 3: Similarity (Jaccard) ===")
    # pairwise_distances with 'jaccard' expects boolean-like
    Xt = (X_test > 0).astype(np.uint8)
    Xr = (X_train > 0).astype(np.uint8)
    D = pairwise_distances(Xt, Xr, metric="jaccard", n_jobs=1)
    S = 1.0 - D
    tau = 1.0 - RADIUS
    print(f"tau = {tau:.2f} (radius={RADIUS})")

    print("\n=== STEP 4: Per-target local RF ===")
    n_test = X_test.shape[0]
    y_pred = np.full(n_test, np.nan)
    n_used = np.zeros(n_test, dtype=int)

    for i in range(n_test):
        nbr_idx = np.where(S[i] >= tau)[0]
        if nbr_idx.size == 0:
            continue
        if K_CAP is not None and nbr_idx.size > K_CAP:
            order = np.argsort(S[i, nbr_idx])[::-1]  # most similar first
            nbr_idx = nbr_idx[order[:K_CAP]]
        n_used[i] = nbr_idx.size

        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
        rf.fit(X_train[nbr_idx], y_train[nbr_idx])
        y_pred[i] = rf.predict(X_test[i:i+1])[0]

    print("\n=== STEP 5: Metrics + outputs ===")
    mask = ~np.isnan(y_pred)
    if mask.sum() > 0:
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        rmse = safe_rmse(y_test[mask], y_pred[mask])
        r2 = r2_score(y_test[mask], y_pred[mask])
        print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R^2={r2:.3f}  (n={int(mask.sum())})")
    else:
        print("No valid predictions (all NaN).")

    # Save CSV
    out_csv = f"{OUT_PREFIX}_rarf_baseline_predictions.csv"
    out = {
        "target_idx": np.arange(n_test, dtype=int),
        "y_true": y_test,
        "y_pred": y_pred,
        "abs_err": np.abs(y_pred - y_test),
        "n_neighbors_used": n_used,
    }
    pred_df = pd.DataFrame(out)

    # attach identifiers for convenience
    for col in ["Imine","Nucleophile","Ligand","Solvent","Temp"]:
        if col in meta_test.columns:
            pred_df[col] = meta_test[col].values

    pred_df.to_csv(out_csv, index=False)
    print(f"→ Wrote {out_csv}")

    # Parity plot
    import matplotlib
    matplotlib.use("Agg")
    plt.figure(figsize=(5,5))
    yt, yp = y_test[mask], y_pred[mask]
    plt.scatter(yt, yp, s=30)
    if yt.size:
        n, m = float(np.min([yt.min(), yp.min()])), float(np.max([yt.max(), yp.max()]))
        plt.plot([n,m], [n,m], "--")
    plt.xlabel("Measured ΔΔG‡")
    plt.ylabel("Predicted ΔΔG‡ (Baseline)")
    plt.title(f"RaRF Baseline Parity (radius={RADIUS}, k_cap={K_CAP})")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_parity_baseline.png", dpi=300)
    print(f"→ Wrote {OUT_PREFIX}_parity_baseline.png")

    # Also write a quick summary JSON
    summary = dict(
        radius=RADIUS, tau=float(tau), k_cap=K_CAP,
        mae=float(mae) if mask.sum()>0 else None,
        rmse=float(rmse) if mask.sum()>0 else None,
        r2=float(r2) if mask.sum()>0 else None,
        n_preds=int(mask.sum())
    )
    with open(f"{OUT_PREFIX}_baseline_summary.json","w") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"→ Wrote {OUT_PREFIX}_baseline_summary.json")
