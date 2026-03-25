"""
==================================================================================
 run_pipeline.py — Master Runner for Telecom Churn Analysis
==================================================================================
 Project  : Customer Churn Analysis for Telecom Industry
 Intern   : Data Analyst Internship — Elevate Labs
 Purpose  : Runs the complete pipeline in sequence:
            1. Generate synthetic dataset
            2. Run SQL aggregation queries
            3. Execute ML pipeline (training, evaluation, segmentation)
            4. Generate HTML report

 Usage:
    python run_pipeline.py

 Note:
    Individual scripts can also be run independently:
    python 01_generate_data.py
    python 02_sql_analysis.py
    python 03_ml_pipeline.py
    python 04_generate_report.py
==================================================================================
"""

import time
import os
import sys

def run_step(step_name, module_path):
    """Run a pipeline step and measure execution time."""
    print(f"\n{'━' * 60}")
    print(f"  🚀 Running: {step_name}")
    print(f"{'━' * 60}")
    
    start = time.time()
    
    # Import and run the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(step_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Set __file__ so relative paths work inside each module
    module.__file__ = module_path
    spec.loader.exec_module(module)
    
    elapsed = time.time() - start
    print(f"\n  ⏱️  Completed in {elapsed:.1f}s")
    return elapsed


def main():
    """Run the full pipeline."""
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  📊 TELECOM CUSTOMER CHURN ANALYSIS — FULL PIPELINE     ║")
    print("║  Data Analyst Internship · Elevate Labs                  ║")
    print("╚" + "═" * 58 + "╝")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    total_start = time.time()
    timings = {}

    # ── Step 1: Generate Data ────────────────────────────────────────────
    timings["01_generate_data"] = run_step(
        "Step 1: Generate Synthetic Dataset",
        os.path.join(base_dir, "01_generate_data.py")
    )

    # ── Step 2: SQL Analysis ─────────────────────────────────────────────
    timings["02_sql_analysis"] = run_step(
        "Step 2: SQL Aggregation Queries",
        os.path.join(base_dir, "02_sql_analysis.py")
    )

    # ── Step 3: ML Pipeline ──────────────────────────────────────────────
    timings["03_ml_pipeline"] = run_step(
        "Step 3: ML Pipeline (EDA + Models + SHAP + Segmentation)",
        os.path.join(base_dir, "03_ml_pipeline.py")
    )

    # ── Step 4: Generate Report ──────────────────────────────────────────
    timings["04_generate_report"] = run_step(
        "Step 4: Generate HTML Report",
        os.path.join(base_dir, "04_generate_report.py")
    )

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  ✅ PIPELINE COMPLETE                                    ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  ⏱️  Total time: {total_time:.1f}s")
    print(f"\n  Step timings:")
    for step, t in timings.items():
        print(f"    {step}: {t:.1f}s")
    
    report_path = os.path.join(base_dir, "outputs", "Telecom_Churn_Analysis_Report.html")
    print(f"\n  📄 Open the report:")
    print(f"     {report_path}")
    print(f"\n  📁 All outputs in:")
    print(f"     {os.path.join(base_dir, 'outputs')}")


if __name__ == "__main__":
    main()
