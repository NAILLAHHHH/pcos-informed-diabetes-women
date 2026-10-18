"""Preprocessing and feature engineering (no synthetic PCOS labels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE

# BRFSS cleaned indicators are coded survey responses; BMI is the main continuous risk of outliers.
BMI_PLAUSIBLE_MIN = 12.0
BMI_PLAUSIBLE_MAX = 60.0


def diagnose_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness, dtypes, and simple outlier flags for documentation/reproducibility."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": missing,
            "missing_pct": 100.0 * missing / n if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if col == "BMI" and pd.api.types.is_numeric_dtype(s):
            row["outlier_low"] = int((s < BMI_PLAUSIBLE_MIN).sum())
            row["outlier_high"] = int((s > BMI_PLAUSIBLE_MAX).sum())
            row["q01"] = float(s.quantile(0.01))
            row["q99"] = float(s.quantile(0.99))
        rows.append(row)
    return pd.DataFrame(rows)


def handle_missing_and_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the women cohort prior to modeling.

    The cleaned BRFSS indicators release is largely complete; this step still:
    - drops rows with missing target or required features
    - winsorizes/clips implausible BMI values to a clinically plausible survey range
    """
    out = df.copy()
    report = {"n_before": int(len(out))}

    required = [TARGET_COL] + [c for c in FEATURE_COLS if c in out.columns]
    missing_mask = out[required].isna().any(axis=1)
    report["rows_dropped_missing"] = int(missing_mask.sum())
    out = out.loc[~missing_mask].copy()

    if "BMI" in out.columns:
        bmi = out["BMI"].astype(float)
        report["bmi_below_min"] = int((bmi < BMI_PLAUSIBLE_MIN).sum())
        report["bmi_above_max"] = int((bmi > BMI_PLAUSIBLE_MAX).sum())
        out["BMI"] = bmi.clip(BMI_PLAUSIBLE_MIN, BMI_PLAUSIBLE_MAX)

    report["n_after"] = int(len(out))
    return out, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transparent metabolic composites motivated by PCOS–diabetes pathways.

    These are engineered from existing survey fields. They are NOT a PCOS diagnosis.
    """
    out = df.copy()

    # Cardiometabolic burden: hypertension + high cholesterol + elevated BMI band
    out["MetabolicBurden"] = (
        out["HighBP"].astype(float)
        + out["HighChol"].astype(float)
        + (out["BMI"] >= 30).astype(float)
    )

    # Low activity + poorer general health (lifestyle / metabolic stress proxy)
    out["LifestyleRisk"] = (
        (1.0 - out["PhysActivity"].astype(float))
        + (out["GenHlth"].astype(float) / 5.0)
    )

    # Interaction often discussed in metabolic syndrome literature
    out["BMI_x_HighBP"] = out["BMI"].astype(float) * out["HighBP"].astype(float)

    return out


def feature_matrix_columns(df: pd.DataFrame, include_engineered: bool = True) -> list[str]:
    engineered = ["MetabolicBurden", "LifestyleRisk", "BMI_x_HighBP"]
    cols = list(FEATURE_COLS)
    if include_engineered:
        cols = cols + engineered
    return [c for c in cols if c in df.columns and c != TARGET_COL]


def stratified_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Return train / val / test frames with stratified target splits."""
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_scaler(feature_cols: list[str]) -> ColumnTransformer:
    """Scale all model features for linear/neural parity; trees tolerate scaling."""
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def iqr_outlier_counts(series: pd.Series) -> dict[str, float]:
    """Classic Tukey IQR fence counts (reported for BMI audit)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "fence_low": float(low),
        "fence_high": float(high),
        "n_below": float((series < low).sum()),
        "n_above": float((series > high).sum()),
    }
