#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare k-matched summaries side-by-side
"""
import json
from pathlib import Path

def main():
    b = json.loads(Path("summary_baseline_k.json").read_text())
    s = json.loads(Path("summary_shared_k.json").read_text())
    comp = {
        "k": b.get("k"),
        "tau": b.get("tau"),
        "baseline": {"MAE": b["mae"], "RMSE": b["rmse"], "R2": b["r2"], "Union": b["union"]},
        "shared": {"MAE": s["mae"], "RMSE": s["rmse"], "R2": s["r2"], "Union": s["union"]},
        "delta": {
            "ΔMAE_shared_minus_base": s["mae"] - b["mae"],
            "Union_ratio_base_over_shared": (b["union"] / max(1, s["union"]))
        }
    }
    Path("compare_k_match.json").write_text(json.dumps(comp, indent=2))
    print(json.dumps(comp, indent=2))

if __name__ == "__main__":
    main()
