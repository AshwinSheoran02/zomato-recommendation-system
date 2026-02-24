"""
🔥 00_make_data.py — One-click data generation pipeline
=========================================================

Runs all data generation steps in sequence:
  1. setup_env.py          → install packages & create folders
  2. 01_generate_base_tables.py → users, restaurants, items
  3. 02_generate_orders.py      → orders, order_items
  4. 03_build_training_table.py → training_rows, baseline

Usage:
    python scripts/00_make_data.py
"""

import time
import importlib
import sys
import pathlib

# Ensure scripts/ is on the path so we can import sibling modules
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

STEPS = [
    ("setup_env",               "🔧 Installing packages & creating folders"),
    ("01_generate_base_tables",  "📋 Generating users / restaurants / items"),
    ("02_generate_orders",       "🛒 Generating orders & order_items"),
    ("03_build_training_table",  "🧠 Building training_rows & baseline"),
]


def run_all():
    print("=" * 65)
    print("  🔥 ZOMATO CSAO — FULL DATA GENERATION PIPELINE")
    print("=" * 65)

    total_start = time.time()

    for module_name, description in STEPS:
        print(f"\n{'─' * 65}")
        print(f"  STEP: {description}")
        print(f"{'─' * 65}\n")

        step_start = time.time()
        module = importlib.import_module(module_name)
        module.main()
        elapsed = time.time() - step_start

        print(f"\n  ⏱️  {description} done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f"  🏁 ALL DONE — Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run_all()
