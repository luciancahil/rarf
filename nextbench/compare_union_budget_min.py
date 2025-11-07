#!/usr/bin/env python3
# Compare union-budget summaries
import json
from pathlib import Path

def main():
    b = json.loads(Path("summary_baseline_B.json").read_text())
    s = json.loads(Path("summary_shared_B.json").read_text())
    comp = {
        "tau": b.get("tau"),
        "budget_B": b.get("budget"),
        "k_report": b.get("k"),
        "baseline_B": {
            "MAE": b["mae"], "RMSE": b["rmse"], "R2": b["r2"],
            "UnionAchieved": b["union"], "%targets>=k": b["pct_targets_ge_k"]
        },
        "shared_B": {
            "MAE": s["mae"], "RMSE": s["rmse"], "R2": s["r2"],
            "UnionAchieved": s["union"], "%targets>=k": s["pct_targets_ge_k"]
        },
        "delta": {
            "ΔMAE_shared_minus_base": s["mae"] - b["mae"],
            "Δ%targets>=k_shared_minus_base": s["pct_targets_ge_k"] - b["pct_targets_ge_k"]
        }
    }
    Path("compare_union_budget_min.json").write_text(json.dumps(comp, indent=2))
    print(json.dumps(comp, indent=2))

if __name__ == "__main__":
    main()
