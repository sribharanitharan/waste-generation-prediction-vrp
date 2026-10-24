"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Master Pipeline Runner — Smart Waste Generation Prediction System
─────────────────────────────────────────────────────────────────────────────
Executes all 6 stages in sequence with timing and error handling.
Usage:
    python run_pipeline.py               # Run all stages
    python run_pipeline.py --from 2      # Resume from stage 2
    python run_pipeline.py --only 3      # Run only stage 3
"""

import subprocess
import sys
import time
import argparse
import os

STAGES = [
    (0, "scripts/00_generate_synthetic_data.py", "Synthetic Data Generation"),
    (1, "scripts/01_spatial_feature_builder.py", "Spatial Feature Engineering"),
    (2, "scripts/02_spatial_cv_models.py",       "Multi-Output ML + Spatial CV"),
    (3, "scripts/03_geo_xai_shap.py",            "Geo-XAI / TreeSHAP Analysis"),
    (4, "scripts/04_prescriptive_routing.py",    "VRP Waste Truck Routing"),
    (5, "scripts/05_carbon_impact_calc.py",      "IPCC Carbon Impact Mapping"),
]

SEP = "=" * 65


def run_stage(script_path: str, stage_name: str) -> bool:
    print(f"\n{SEP}", flush=True)
    print(f"  STAGE: {stage_name}", flush=True)
    print(f"  Script: {script_path}", flush=True)
    print(SEP, flush=True)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "4"
    result = subprocess.run(
        [sys.executable, "-u", script_path],
        capture_output=False,
        env=env
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n  [ERROR] Stage failed with exit code {result.returncode}", flush=True)
        return False
    print(f"\n  [OK] Completed in {elapsed:.1f}s", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Smart Waste Pipeline Runner")
    parser.add_argument("--from", dest="from_stage", type=int, default=0,
                        help="Start from stage N (0-indexed)")
    parser.add_argument("--only", dest="only_stage", type=int, default=None,
                        help="Run only stage N")
    args = parser.parse_args()

    print(SEP)
    print("  Smart Waste Generation Prediction & Route Optimization")
    print("  End-to-End Geospatial ML Pipeline  |  Bengaluru, India")
    print(SEP)

    stages_to_run = STAGES
    if args.only_stage is not None:
        stages_to_run = [(n, s, name) for n, s, name in STAGES if n == args.only_stage]
    elif args.from_stage > 0:
        stages_to_run = [(n, s, name) for n, s, name in STAGES if n >= args.from_stage]

    total_start = time.perf_counter()
    for stage_num, script, name in stages_to_run:
        ok = run_stage(script, f"[{stage_num}] {name}")
        if not ok:
            print(f"\n  Pipeline aborted at stage {stage_num}.")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{SEP}")
    print(f"  ALL STAGES COMPLETE in {total_elapsed:.1f}s")
    print(f"\n  Outputs:")
    print(f"    data/processed/fused_spatial_features.gpkg")
    print(f"    data/processed/final_predictions_with_shap.gpkg")
    print(f"    data/processed/optimized_routes.gpkg")
    print(f"    data/processed/carbon_impact.gpkg")
    print(f"    maps/  (6 publication-quality PNG maps)")
    print(f"    report/carbon_impact_summary.csv")
    print(f"\n  Load .gpkg files in QGIS for interactive exploration!")
    print(SEP)


if __name__ == "__main__":
    main()
