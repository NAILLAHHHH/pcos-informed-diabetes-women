"""Load and validate the BRFSS diabetes indicators dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATA_RAW,
    FEMALE_CODE,
    FEATURE_COLS,
    RAW_FILENAME,
    SEX_COL,
    TARGET_COL,
)


def resolve_raw_path(raw_path: str | Path | None = None) -> Path:
    path = Path(raw_path) if raw_path else DATA_RAW / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Run `python scripts/download_data.py` or place the CSV in data/raw/."
        )
    return path


def load_raw_dataset(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load the full raw CSV (must be ≥ 20,000 rows before filtering)."""
    path = resolve_raw_path(raw_path)
    df = pd.read_csv(path)
    if len(df) < 20_000:
        raise ValueError(
            f"Raw dataset has {len(df)} rows; assignment requires ≥ 20,000 before preprocessing."
        )
    return df


def filter_women(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict analysis to female respondents for the project's clinical framing."""
    if SEX_COL not in df.columns:
        raise KeyError(f"Expected column '{SEX_COL}' for sex filtering.")
    women = df.loc[df[SEX_COL] == FEMALE_CODE].copy()
    if len(women) < 20_000:
        raise ValueError(
            f"Female subset has only {len(women)} rows; check Sex encoding or source file."
        )
    return women


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in [TARGET_COL, *FEATURE_COLS] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_women_diabetes_frame(raw_path: str | Path | None = None) -> pd.DataFrame:
    """End-to-end load: raw CSV → validate → female subset."""
    df = load_raw_dataset(raw_path)
    validate_columns(df)
    women = filter_women(df)
    # Sex is used only for cohort definition; drop to avoid trivial leakage of filter
    return women.drop(columns=[SEX_COL])
