"""
🔧 Environment Setup — Zomato CSAO Recommendation System
Installs required packages and creates the folder structure.
"""

import subprocess
import sys
import pathlib


def main():
    # ── Install packages ──────────────────────────────────────────────
    packages = [
        "numpy<2", "pandas>=2.0", "faker", "tqdm",
        "scikit-learn", "lightgbm", "matplotlib",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All packages installed.")

    # ── Create folder structure ───────────────────────────────────────
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    RAW  = ROOT / "data" / "raw"
    PROC = ROOT / "data" / "processed"

    for d in [RAW, PROC]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"📁 {d}")

    print("\n✅ Folder structure ready.")


if __name__ == "__main__":
    main()
