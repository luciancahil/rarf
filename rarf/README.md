
# RaRF Shared vs Baseline (Comparison Suite)

## What this does
- Builds Morgan fingerprints from `Nature_SMILES.xlsx` (uses the per-part lookup sheets).
- Runs **RaRF-Shared** (greedy shared training selection + per-target enrichment).
- Runs **Vanilla RaRF** baseline (per-target neighbors only, no sharing).
- Writes **CSV outputs** and **plots** for both, plus **budget sweeps**.

## Files
- `utils.py` – SMILES→Morgan bits helpers
- `RaRFRegressor_shared_overlap_visual.py` – RaRF-Shared implementation
- `run_rarf_compare.py` – main entry (runs once + sweeps budgets)
- `requirements.txt` – minimal Python deps

## Setup (conda suggested)
> RDKit wheels compiled against NumPy 1.x. Use numpy<2.0.

```bash
conda create -n rarfviz python=3.10 -y
conda activate rarfviz
pip install -r requirements.txt
# if you hit RDKit/NumPy _ARRAY_API issues:
pip install "numpy<2.0" --force-reinstall
```

## Run
Place `Nature_SMILES.xlsx` in the same folder and run:

```bash
python run_rarf_compare.py
```

## Outputs
- `single_rarf_compare_shared_vs_baseline.csv` – y_true, y_pred_shared, y_pred_baseline, errors, and metadata
- `single_parity_shared.png` / `single_parity_baseline.png` – parity plots
- `single_summary.json` – quick metrics summary (MAE/RMSE/R2, coverage, etc.)
- `mae_vs_budget.csv` + `mae_vs_budget.png` – MAE vs. shared budget curve (baseline shown as flat line)
- `coverage_vs_budget.png` – fraction of test targets with ≥1 shared neighbor vs. budget
