#!/usr/bin/env python3
"""Download the BRFSS diabetes health indicators CSV into data/raw/ (not committed)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET_NAME = "diabetes_012_health_indicators_BRFSS2015.csv"


def download_via_kagglehub() -> Path:
    import kagglehub

    print("Downloading via kagglehub: alexteboul/diabetes-health-indicators-dataset ...")
    dataset_dir = Path(
        kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    )
    matches = list(dataset_dir.rglob(TARGET_NAME))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {TARGET_NAME} under {dataset_dir}. "
            f"Found: {[p.name for p in dataset_dir.rglob('*.csv')]}"
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / TARGET_NAME
    shutil.copy2(matches[0], dest)
    return dest


def main() -> int:
    dest = RAW_DIR / TARGET_NAME
    if dest.exists():
        print(f"Already present: {dest} ({dest.stat().st_size:,} bytes)")
        return 0
    try:
        path = download_via_kagglehub()
    except Exception as exc:  # noqa: BLE001 — surface clear next steps for students
        print("Automatic download failed:", exc)
        print(
            "\nManual fallback:\n"
            "1) Download the Diabetes Health Indicators dataset from Kaggle "
            "(or the CDC BRFSS-derived release you prefer).\n"
            f"2) Place `{TARGET_NAME}` in `{RAW_DIR}`.\n"
            "3) Re-run this script to verify the file is present.\n"
        )
        return 1

    print(f"Saved raw dataset to {path}")
    # Quick size check without pandas dependency for script lightness
    with path.open("rb") as f:
        n_lines = sum(1 for _ in f)
    print(f"Line count (including header): {n_lines:,}")
    if n_lines - 1 < 20_000:
        print("WARNING: fewer than 20,000 data rows detected.")
        return 2
    print("OK: raw row count meets the ≥ 20,000 assignment threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
