#!/usr/bin/env python3
"""Run the ML vs DL experiment suite and persist metrics, figures, and models."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET_COL
from src.data_loading import load_raw_dataset, load_women_diabetes_frame
from src.deep_learning import (
    build_mlp,
    compile_model,
    compute_class_weights,
    default_callbacks,
    make_tf_dataset,
    save_keras_model,
    set_tf_seed,
)
from src.evaluation import (
    metrics_table,
    plot_class_balance,
    plot_confusion,
    plot_learning_curve,
    plot_roc_ovr,
    summarize_classification,
)
from src.preprocessing import engineer_features, feature_matrix_columns, stratified_splits
from src.traditional_ml import build_pipelines, save_sklearn_model, tune_model


def main() -> int:
    results_dir = ROOT / "report" / "experiment_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    print(f"Raw shape: {raw.shape}")
    women = load_women_diabetes_frame()
    print(f"Women cohort: {women.shape}")
    plot_class_balance(
        women[TARGET_COL],
        title="Diabetes class balance (women cohort)",
        filename="class_balance_women.png",
    )
    plt.close("all")

    df = engineer_features(women)
    feature_cols = feature_matrix_columns(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df)
    X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]
    X_tune = pd.concat([X_train, X_val])
    y_tune = pd.concat([y_train, y_val])
    print("Split sizes:", X_train.shape, X_val.shape, X_test.shape)

    # --- Traditional ML ---
    # Skip full linear_svm on the entire set in the first automated run (slow);
    # keep it available in the notebook for the formal experiment write-up.
    pipelines = build_pipelines(feature_cols)
    pipelines.pop("linear_svm", None)

    fitted = {}
    rows = []
    for name, pipe in pipelines.items():
        print(f"\n=== Baseline {name} ===")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        summary = summarize_classification(y_test, pred, proba, title=f"{name} (baseline)")
        rows.append(summary)
        fitted[name] = pipe
        print({k: summary[k] for k in ("accuracy", "macro_f1", "roc_auc_ovr_macro") if k in summary})

    for name in ["log_reg", "random_forest", "grad_boost"]:
        print(f"\n=== Tuning {name} ===")
        search = tune_model(name, build_pipelines(feature_cols)[name], X_tune, y_tune, n_iter=6, cv=3)
        best = search.best_estimator_
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)
        summary = summarize_classification(y_test, pred, proba, title=f"{name} (tuned)")
        rows.append(summary)
        fitted[f"{name}_tuned"] = best
        print("Best params:", search.best_params_)
        print({k: summary[k] for k in ("accuracy", "macro_f1", "roc_auc_ovr_macro") if k in summary})

    best_trad_key, best_trad_f1 = max(
        (
            (k, f1_score(y_test, m.predict(X_test), average="macro", zero_division=0))
            for k, m in fitted.items()
        ),
        key=lambda t: t[1],
    )
    best_trad = fitted[best_trad_key]
    print(f"\nBest traditional: {best_trad_key} | macro-F1={best_trad_f1:.4f}")
    trad_pred = best_trad.predict(X_test)
    trad_proba = best_trad.predict_proba(X_test)
    plot_confusion(y_test, trad_pred, f"Confusion — {best_trad_key}", "cm_best_traditional.png")
    plot_roc_ovr(y_test, trad_proba, f"ROC OvR — {best_trad_key}", "roc_best_traditional.png")
    plt.close("all")

    # Learning curve on a subsample for runtime
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_tune), size=min(25_000, len(X_tune)), replace=False)
    plot_learning_curve(
        build_pipelines(feature_cols)["random_forest"],
        X_tune.iloc[idx],
        y_tune.iloc[idx],
        title="Learning curve — Random Forest (macro-F1, 25k subsample)",
        filename="learning_curve_rf.png",
        cv=3,
    )
    plt.close("all")

    # --- Deep learning ---
    set_tf_seed(RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    ds_train = make_tf_dataset(X_train_s, y_train.to_numpy(), batch_size=256, shuffle=True)
    ds_val = make_tf_dataset(X_val_s, y_val.to_numpy(), batch_size=256, shuffle=False)
    class_weight = compute_class_weights(y_train.to_numpy())

    configs = [
        {"name": "mlp_128_64_32", "hidden": (128, 64, 32), "dropout": 0.3, "lr": 1e-3},
        {"name": "mlp_64_32", "hidden": (64, 32), "dropout": 0.2, "lr": 1e-3},
        {"name": "mlp_256_128_64", "hidden": (256, 128, 64), "dropout": 0.4, "lr": 5e-4},
    ]
    dl_models = {}
    histories = {}
    for cfg in configs:
        print(f"\n=== DL {cfg['name']} ===")
        model = compile_model(
            build_mlp(
                n_features=X_train_s.shape[1],
                hidden_units=cfg["hidden"],
                dropout=cfg["dropout"],
            ),
            learning_rate=cfg["lr"],
        )
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=40,
            class_weight=class_weight,
            callbacks=default_callbacks(patience=7),
            verbose=2,
        )
        proba = model.predict(X_test_s, verbose=0)
        pred = proba.argmax(axis=1)
        summary = summarize_classification(y_test, pred, proba, title=cfg["name"])
        rows.append(summary)
        dl_models[cfg["name"]] = model
        histories[cfg["name"]] = {k: [float(x) for x in v] for k, v in history.history.items()}
        print({k: summary[k] for k in ("accuracy", "macro_f1", "roc_auc_ovr_macro") if k in summary})

    best_dl_name = max(
        dl_models,
        key=lambda n: f1_score(
            y_test,
            dl_models[n].predict(X_test_s, verbose=0).argmax(axis=1),
            average="macro",
            zero_division=0,
        ),
    )
    best_dl = dl_models[best_dl_name]
    dl_proba = best_dl.predict(X_test_s, verbose=0)
    dl_pred = dl_proba.argmax(axis=1)
    plot_confusion(y_test, dl_pred, f"Confusion — {best_dl_name}", "cm_best_dl.png")
    plot_roc_ovr(y_test, dl_proba, f"ROC OvR — {best_dl_name}", "roc_best_dl.png")

    hist = histories[best_dl_name]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title(f"Loss — {best_dl_name}")
    axes[0].legend()
    axes[1].plot(hist["accuracy"], label="train")
    axes[1].plot(hist["val_accuracy"], label="val")
    axes[1].set_title(f"Accuracy — {best_dl_name}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "dl_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    table = metrics_table(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(results_dir / "metrics_summary.csv", index=False)
    print("\n=== Metrics summary ===")
    print(table.to_string(index=False))

    trad_path = save_sklearn_model(best_trad, "best_traditional_ml.joblib")
    joblib.dump(scaler, ROOT / "models" / "dl_standard_scaler.joblib")
    dl_path = save_keras_model(best_dl, "best_deep_learning.keras")

    meta = {
        "raw_rows": int(len(raw)),
        "women_rows": int(len(women)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "best_traditional": best_trad_key,
        "best_traditional_macro_f1": float(best_trad_f1),
        "best_deep_learning": best_dl_name,
        "best_deep_learning_macro_f1": float(
            f1_score(y_test, dl_pred, average="macro", zero_division=0)
        ),
        "model_paths": {"traditional": str(trad_path), "deep_learning": str(dl_path)},
        "histories": histories,
    }
    (results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved models:", trad_path, dl_path)
    print("Saved metrics:", results_dir / "metrics_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
