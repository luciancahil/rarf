
import numpy as np
import pandas as pd
import time, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from RaRFRegressor_shared_overlap_visual import RaRFRegressor
from utils import smi2morgan

def log_section(title):
    print("\n" + "="*80)
    print(f"🔷 {title}")
    print("="*80)

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

def to_int_list(seq):
    return [int(x) for x in seq]

def safe_rmse(y_true, y_pred):
    # Support both new and old scikit-learn APIs
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

try:
    start_total = time.time()

    log_section("STEP 1: LOADING EXCEL & BUILDING FPs")
    X, y, meta = build_X_from_excel("Nature_SMILES.xlsx", nbits=2048)
    print(f"   X shape: {X.shape} | y: {y.shape}")

    log_section("STEP 2: TRAIN/TEST SPLIT (80/20)")
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, train_size=0.8, random_state=42
    )
    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

    log_section("STEP 3: INITIALIZING RaRF")
    rarf = RaRFRegressor(radius=0.4, metric="jaccard")
    print(f"   radius={rarf.radius}, metric='{rarf.metric}'")

    log_section("STEP 4: SELECTION + PREDICTION (enrich per-target)")
    out = rarf.fit_predict_shared(
        X_train, y_train, X_test,
        budget=30, alpha=1.2, redundancy_lambda=0.1,
        enrich_min_per_target=5, enrich_from="neighbors", enrich_max_extra=10
    )
    tau = out["tau"]
    sim = out["sim"]
    selected = np.array(out["selected"], dtype=int)
    local_sets = out["local_sets"]
    y_pred = out["preds"]

    # Coverage audit
    rows = []
    for i in range(sim.shape[0]):
        nbrs_all = np.where(sim[i] >= tau)[0]
        nbrs_sel = [j for j in nbrs_all if j in selected]
        rows.append({
            "target_idx": int(i),
            "n_neighbors_all": int(len(nbrs_all)),
            "n_neighbors_selected": int(len(nbrs_sel)),
            "neighbors_all_idx": json.dumps(to_int_list(nbrs_all)),
            "neighbors_selected_idx": json.dumps(to_int_list(nbrs_sel))
        })
    coverage_df = pd.DataFrame(rows)
    coverage_df.to_csv("rarf_coverage_report.csv", index=False)

    # Enrichment detail
    enrich_rows = []
    for i in range(sim.shape[0]):
        final_set = local_sets[i]
        from_shared = [j for j in final_set if j in selected]
        added = [j for j in final_set if j not in selected]
        enrich_rows.append({
            "target_idx": int(i),
            "final_set_size": int(len(final_set)),
            "n_from_shared": int(len(from_shared)),
            "n_added": int(len(added)),
            "added_indices": json.dumps(to_int_list(added))
        })
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv("rarf_enriched_training_sets.csv", index=False)

    log_section("STEP 5: MEASURED vs PREDICTED CSV + PLOTS")

    # Build per-target summary (test fold only)
    pred_df = pd.DataFrame({
        "target_idx": np.arange(len(y_test), dtype=int),
        "y_true": y_test,
        "y_pred": y_pred,
    })
    pred_df["abs_err"] = np.abs(pred_df["y_pred"] - pred_df["y_true"])
    # attach coverage + training sizes
    pred_df = pred_df.merge(coverage_df[["target_idx","n_neighbors_all","n_neighbors_selected"]], on="target_idx", how="left")
    pred_df = pred_df.merge(enrich_df[["target_idx","final_set_size","n_from_shared","n_added"]], on="target_idx", how="left")
    # include identifying metadata if helpful
    for col in ["Imine","Nucleophile","Ligand","Solvent","Temp"]:
        if col in meta_test.columns:
            pred_df[col] = meta_test[col].values

    pred_df.to_csv("rarf_predictions.csv", index=False)
    print("   ✅ Wrote rarf_predictions.csv")

    # Metrics (drop NaNs first)
    mask = ~np.isnan(y_pred)
    if mask.sum() > 0:
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        rmse = safe_rmse(y_test[mask], y_pred[mask])
        r2 = r2_score(y_test[mask], y_pred[mask])
        print(f"   Metrics (on {int(mask.sum())}/{len(mask)} preds): MAE={mae:.3f} | RMSE={rmse:.3f} | R^2={r2:.3f}")
    else:
        print("   Metrics: no valid predictions (all NaN).")

    # Parity plot
    plt.figure(figsize=(5,5))
    yt, yp = y_test[mask], y_pred[mask]
    plt.scatter(yt, yp, s=30)
    if yt.size > 0:
        m = max(np.max(yt), np.max(yp))
        n = min(np.min(yt), np.min(yp))
        plt.plot([n, m], [n, m], linestyle="--")
    plt.xlabel("Measured ΔΔG‡")
    plt.ylabel("Predicted ΔΔG‡")
    plt.title("Parity Plot (Test Fold)")
    plt.tight_layout()
    plt.savefig("parity.png", dpi=300)

    # Error histogram
    plt.figure(figsize=(5,4))
    if mask.sum() > 0:
        plt.hist(np.abs(yp - yt), bins=20)
    plt.xlabel("|Error| (ΔΔG‡)")
    plt.ylabel("Count")
    plt.title("Absolute Error Histogram")
    plt.tight_layout()
    plt.savefig("error_hist.png", dpi=300)

    log_section("STEP 6: UMAP PLOT")
    rarf.plot_overlap_map(X_train, X_test, selected)
    print("   ✅ Saved 'overlap_map.png'")

    log_section("STEP 7: SUMMARY")
    covered = (sim[:, selected] >= tau).any(axis=1) if selected.size>0 else np.zeros(sim.shape[0], dtype=bool)
    print(f"   Shared coverage (≥1 selected neighbor): {covered.sum()}/{sim.shape[0]}")
    print(f"   Predictions CSV: rarf_predictions.csv")
    print(f"   Parity plot: parity.png  |  Error histogram: error_hist.png")
    print(f"   Total runtime: {time.time()-start_total:.2f}s")

except Exception as e:
    import traceback
    print("\n" + "="*80 + "\n❌ ERROR OCCURRED\n" + "="*80)
    traceback.print_exc()
    print("\nIf you see an RDKit/Numpy _ARRAY_API error: pip install 'numpy<2.0' --force-reinstall")
