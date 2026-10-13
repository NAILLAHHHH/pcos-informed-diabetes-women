"""Preprocessing and feature engineering (no synthetic PCOS labels)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE


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


def feature_matrix_columns(df: pd.DataFrame) -> list[str]:
    engineered = ["MetabolicBurden", "LifestyleRisk", "BMI_x_HighBP"]
    return [c for c in FEATURE_COLS + engineered if c in df.columns and c != TARGET_COL]


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
    """Scale continuous-ish columns; pass through binary indicators unchanged if desired.

    For simplicity and reproducibility we scale all model features. Tree models can
    ignore scaling; linear / neural models benefit from it.
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
