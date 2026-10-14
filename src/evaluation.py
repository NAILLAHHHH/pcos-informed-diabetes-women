"""Shared evaluation helpers: metrics, plots, learning curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import LearningCurveDisplay
from sklearn.preprocessing import label_binarize

from src.config import CLASS_NAMES, FIGURES_DIR


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def multiclass_roc_auc(y_true, y_proba, labels=(0, 1, 2)) -> float:
    y_bin = label_binarize(y_true, classes=list(labels))
    return float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))


def summarize_classification(y_true, y_pred, y_proba=None, title: str = "Model") -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {
        "title": title,
        "accuracy": report["accuracy"],
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "report": report,
    }
    if y_proba is not None:
        out["roc_auc_ovr_macro"] = multiclass_roc_auc(y_true, y_proba)
    return out


def plot_confusion(y_true, y_pred, title: str, filename: str | None = None):
    ensure_figures_dir()
    labels = sorted(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[CLASS_NAMES[i] for i in labels],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_ovr(y_true, y_proba, title: str, filename: str | None = None):
    ensure_figures_dir()
    labels = sorted(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, cls in enumerate(labels):
        RocCurveDisplay.from_predictions(
            y_bin[:, i],
            y_proba[:, i],
            name=CLASS_NAMES[cls],
            ax=ax,
        )
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    fig.tight_layout()
    if filename:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curve(estimator, X, y, title: str, filename: str | None = None, cv=3):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "title",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc_ovr_macro",
    ]
    frame = pd.DataFrame(rows)
    for c in cols:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame[cols]


def plot_class_balance(y, title: str = "Class balance", filename: str | None = None):
    ensure_figures_dir()
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=[CLASS_NAMES[i] for i in counts.index],
        y=counts.values,
        ax=ax,
        color="#2F6F8F",
    )
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig
