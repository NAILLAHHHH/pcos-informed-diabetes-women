"""Traditional scikit-learn classifiers and tuning helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import MODELS_DIR, N_JOBS, RANDOM_STATE
from src.preprocessing import make_scaler


def build_pipelines(feature_cols: list[str]) -> dict[str, Pipeline]:
    scaler = make_scaler(feature_cols)
    return {
        "log_reg": Pipeline(
            steps=[
                ("prep", scaler),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", make_scaler(feature_cols)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                    ),
                ),
            ]
        ),
        "grad_boost": Pipeline(
            steps=[
                ("prep", make_scaler(feature_cols)),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("prep", make_scaler(feature_cols)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            max_iter=5000,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def tuning_spaces() -> dict[str, dict]:
    return {
        "log_reg": {
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "random_forest": {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [8, 12, 16, None],
            "clf__min_samples_leaf": [1, 2, 5],
        },
        "grad_boost": {
            "clf__n_estimators": [100, 150, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [2, 3, 4],
        },
        "linear_svm": {
            "clf__estimator__C": [0.1, 0.5, 1.0, 2.0],
        },
    }


def tune_model(
    name: str,
    pipeline: Pipeline,
    X,
    y,
    n_iter: int = 8,
    cv: int = 3,
) -> RandomizedSearchCV:
    spaces = tuning_spaces()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=spaces[name],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
    )
    search.fit(X, y)
    return search


def save_sklearn_model(model, filename: str = "best_traditional_ml.joblib") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path
