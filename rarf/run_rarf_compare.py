
import numpy as np
import pandas as pd
import time, json, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from RaRFRegressor_shared_overlap_visual import RaRFRegressor
from utils import smi2morgan

def safe_rmse(y_true, y_pred):
    try:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(y_true, y_pred))

def log(h):
    print("\n" + "="*80 + f"\n{h}\n" + "="*80)

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

    X = np.hstack([nu, li, im, so])
    y = df["DDG"].values
    meta = df.copy()
    return X, y, meta

def rarf_baseline_predict(X_train, y_train, X_test, sim, tau, k_cap=None, n_estimators=200, random_state=42):
    n_test = X_test.shape[0]
    yb = np.full(n_test, np.nan)
    for i in range(n_test):
        nbr_idx = np.where(sim[i] >= tau)[0]
        if nbr_idx.size == 0:
            continue
        if k_cap is not None and nbr_idx.size > k_cap:
            order = np.argsort(sim[i, nbr_idx])[::-1]
            nbr_idx = nbr_idx[order[:k_cap]]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_train[nbr_idx], y_train[nbr_idx])
        yb[i] = rf.predict(X_test[i:i+1])[0]
    return yb

def run_once(xls_path="Nature_SMILES.xlsx",
             radius=0.4, budget=30,
             enrich_min=5, enrich_extra=10,
             baseline_kcap=None,
             seed=42, out_prefix="run"):
    X, y, meta = build_X_from_excel(xls_path, nbits=2048)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, train_size=0.8, random_state=seed
    )

    # Shared RaRF
    rarf = RaRFRegressor(radius=radius, metric="jaccard", n_estimators=200, random_state=seed)
    out = rarf.fit_predict_shared(
        X_train, y_train, X_test,
        budget=budget, alpha=1.2, redundancy_lambda=0.1,
        enrich_min_per_target=enrich_min, enrich_from="neighbors", enrich_max_extra=enrich_extra
    )
    y_pred_shared = out["preds"]
    sim = out["sim"]; tau = out["tau"]; selected = out["selected"]
    # Baseline RaRF (per-target only)
    y_pred_base = rarf_baseline_predict(X_train, y_train, X_test, sim, tau, k_cap=baseline_kcap, n_estimators=200, random_state=seed)

    # Metrics (drop NaNs when computing)
    mask_s = ~np.isnan(y_pred_shared)
    mask_b = ~np.isnan(y_pred_base)

    def metrics(y_true, y_pred, mask):
        if mask.sum() == 0:
            return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, n=0)
        return dict(MAE=mean_absolute_error(y_true[mask], y_pred[mask]),
                    RMSE=safe_rmse(y_true[mask], y_pred[mask]),
                    R2=r2_score(y_true[mask], y_pred[mask]),
                    n=int(mask.sum()))

    mS = metrics(y_test, y_pred_shared, mask_s)
    mB = metrics(y_test, y_pred_base,   mask_b)

    # Write predictions CSV
    pred_df = pd.DataFrame({
        "target_idx": np.arange(len(y_test), dtype=int),
        "y_true": y_test,
        "y_pred_shared": y_pred_shared,
        "y_pred_baseline": y_pred_base,
        "abs_err_shared": np.abs(y_pred_shared - y_test),
        "abs_err_baseline": np.abs(y_pred_base - y_test),
    })
    # include identifiers
    for col in ["Imine","Nucleophile","Ligand","Solvent","Temp"]:
        if col in meta_test.columns:
            pred_df[col] = meta_test[col].values
    pred_csv = f"{out_prefix}_rarf_compare_shared_vs_baseline.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Parity plot (shared)
    mask = ~np.isnan(y_pred_shared)
    plt.figure(figsize=(5,5))
    yt, yp = y_test[mask], y_pred_shared[mask]
    plt.scatter(yt, yp, s=30)
    if yt.size:
        n, m = float(np.min([yt.min(), yp.min()])), float(np.max([yt.max(), yp.max()]))
        plt.plot([n,m],[n,m],'--')
    plt.xlabel("Measured ΔΔG‡")
    plt.ylabel("Predicted ΔΔG‡ (Shared)")
    plt.title(f"Parity (radius={radius}, budget={budget})")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_parity_shared.png", dpi=300); plt.close()

    # Parity plot (baseline)
    mask = ~np.isnan(y_pred_base)
    plt.figure(figsize=(5,5))
    yt, yp = y_test[mask], y_pred_base[mask]
    plt.scatter(yt, yp, s=30)
    if yt.size:
        n, m = float(np.min([yt.min(), yp.min()])), float(np.max([yt.max(), yp.max()]))
        plt.plot([n,m],[n,m],'--')
    plt.xlabel("Measured ΔΔG‡")
    plt.ylabel("Predicted ΔΔG‡ (Baseline)")
    plt.title(f"Parity (baseline; radius={radius})")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_parity_baseline.png", dpi=300); plt.close()

    # Coverage stats
    selected_arr = np.array(selected, dtype=int)
    covered = (sim[:, selected_arr] >= tau).any(axis=1) if selected_arr.size>0 else np.zeros(sim.shape[0], dtype=bool)
    coverage = covered.sum()

    # Save small summary JSON
    summary = {
        "radius": radius, "budget": budget, "tau": float(tau),
        "metrics_shared": mS, "metrics_baseline": mB,
        "coverage_shared_targets": int(coverage), "n_test": int(sim.shape[0]),
        "n_selected": int(selected_arr.size),
        "pred_csv": pred_csv
    }
    with open(f"{out_prefix}_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    return summary

def sweep_budgets(xls_path="Nature_SMILES.xlsx", budgets=(10,20,30,60,90), radius=0.4, seed=42):
    rows = []
    for b in budgets:
        summ = run_once(xls_path, radius=radius, budget=b, out_prefix=f"b{b}_r{radius}", seed=seed)
        ms, mb = summ["metrics_shared"], summ["metrics_baseline"]
        rows.append({
            "budget": b,
            "MAE_shared": ms["MAE"], "RMSE_shared": ms["RMSE"], "R2_shared": ms["R2"], "n_shared": ms["n"],
            "MAE_baseline": mb["MAE"], "RMSE_baseline": mb["RMSE"], "R2_baseline": mb["R2"], "n_baseline": mb["n"],
            "coverage_shared": summ["coverage_shared_targets"], "n_test": summ["n_test"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("mae_vs_budget.csv", index=False)

    # Plot MAE vs budget
    plt.figure(figsize=(6,4))
    plt.plot(df["budget"], df["MAE_shared"], marker="o", label="Shared")
    plt.hlines(df["MAE_baseline"].iloc[-1], xmin=min(df["budget"]), xmax=max(df["budget"]), linestyles="--", label="Baseline (flat)")
    plt.xlabel("Budget (selected shared training points)")
    plt.ylabel("MAE (ΔΔG‡)")
    plt.title(f"MAE vs Budget (radius={radius})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mae_vs_budget.png", dpi=300); plt.close()

    # Coverage vs budget
    plt.figure(figsize=(6,4))
    cov = (df["coverage_shared"] / df["n_test"]) * 100.0
    plt.plot(df["budget"], cov, marker="o")
    plt.xlabel("Budget")
    plt.ylabel("Coverage of test targets (%)")
    plt.title(f"Shared coverage vs Budget (radius={radius})")
    plt.tight_layout()
    plt.savefig("coverage_vs_budget.png", dpi=300); plt.close()

if __name__ == "__main__":
    log("STEP 1: Load + Featurize")
    X, y, meta = build_X_from_excel("Nature_SMILES.xlsx", nbits=2048)
    print(f"Shapes: X={X.shape}, y={y.shape}")

    log("STEP 2: Single run (default radius=0.4, budget=30)")
    summary = run_once(xls_path="Nature_SMILES.xlsx", radius=0.4, budget=30, out_prefix="single")

    print("Single-run summary:")
    print(summary)

    log("STEP 3: Sweep budgets")
    sweep_budgets("Nature_SMILES.xlsx", budgets=(10,20,30,60,90), radius=0.4, seed=42)

    print("\nArtifacts written:")
    print("  - single_rarf_compare_shared_vs_baseline.csv")
    print("  - single_parity_shared.png, single_parity_baseline.png")
    print("  - single_summary.json")
    print("  - mae_vs_budget.csv, mae_vs_budget.png")
    print("  - coverage_vs_budget.png")
