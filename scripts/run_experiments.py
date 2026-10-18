#!/usr/bin/env python3
"""Run ML vs DL experiments with W&B offline logs and a replicable experiment table."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

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
from src.experiment_logging import (
    EXPERIMENT_TABLE_PATH,
    append_experiment_row,
    base_split_config,
    finish_run,
    init_run,
    log_metrics,
)
from src.preprocessing import (
    diagnose_data_quality,
    engineer_features,
    feature_matrix_columns,
    handle_missing_and_outliers,
    iqr_outlier_counts,
    stratified_splits,
)
from src.traditional_ml import build_pipelines, save_sklearn_model, tune_model


def _reset_experiment_table() -> None:
    if EXPERIMENT_TABLE_PATH.exists():
        EXPERIMENT_TABLE_PATH.unlink()


def _log_experiment(
    *,
    run_id: str,
    approach: str,
    model_name: str,
    hyperparameters: dict,
    split_cfg: dict,
    summary: dict,
    observation: str,
    stage: str,
) -> None:
    metrics = {k: summary[k] for k in summary if k != "report" and not isinstance(summary[k], dict)}
    run = init_run(
        name=run_id,
        config={
            "approach": approach,
            "model_name": model_name,
            "stage": stage,
            "hyperparameters": hyperparameters,
            **split_cfg,
        },
        tags=[approach, stage],
    )
    log_metrics(metrics)
    run.summary.update(metrics)
    finish_run()

    append_experiment_row(
        {
            "run_id": run_id,
            "stage": stage,
            "approach": approach,
            "model": model_name,
            "hyperparameters": json.dumps(hyperparameters),
            "dataset_split": split_cfg["split"],
            "n_train": split_cfg["n_train"],
            "n_val": split_cfg["n_val"],
            "n_test": split_cfg["n_test"],
            "accuracy": metrics.get("accuracy"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
            "observation": observation,
            "wandb_mode": "offline",
        }
    )


def main() -> int:
    results_dir = ROOT / "report" / "experiment_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _reset_experiment_table()

    raw = load_raw_dataset()
    print(f"Raw shape: {raw.shape}")
    women = load_women_diabetes_frame()
    print(f"Women cohort (pre-clean): {women.shape}")

    quality = diagnose_data_quality(women)
    quality.to_csv(results_dir / "data_quality_audit.csv", index=False)
    women_clean, clean_report = handle_missing_and_outliers(women)
    if "BMI" in women_clean.columns:
        clean_report["bmi_iqr"] = iqr_outlier_counts(women_clean["BMI"])
    (results_dir / "cleaning_report.json").write_text(json.dumps(clean_report, indent=2))
    print("Cleaning report:", clean_report)

    plot_class_balance(
        women_clean[TARGET_COL],
        title="Diabetes class balance (women cohort)",
        filename="class_balance_women.png",
    )
    plt.close("all")

    df = engineer_features(women_clean)
    feature_cols = feature_matrix_columns(df, include_engineered=True)
    feature_cols_base = feature_matrix_columns(df, include_engineered=False)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df)
    split_cfg = base_split_config(len(X_train), len(X_val), len(X_test))
    print("Split sizes:", split_cfg)

    X_train_f = X_train[feature_cols]
    X_val_f = X_val[feature_cols]
    X_test_f = X_test[feature_cols]
    X_tune = pd.concat([X_train_f, X_val_f])
    y_tune = pd.concat([y_train, y_val])

    # --- Ablation: with vs without engineered features (log_reg) ---
    for label, cols, stage in [
        ("log_reg_no_engineered", feature_cols_base, "ablation"),
        ("log_reg_with_engineered", feature_cols, "ablation"),
    ]:
        pipe = build_pipelines(cols)["log_reg"]
        pipe.fit(X_train[cols], y_train)
        pred = pipe.predict(X_test[cols])
        proba = pipe.predict_proba(X_test[cols])
        summary = summarize_classification(y_test, pred, proba, title=label)
        obs = (
            "Baseline features only — tests whether metabolic composites add signal."
            if "no_engineered" in label
            else "Engineered MetabolicBurden/LifestyleRisk/BMI_x_HighBP included."
        )
        _log_experiment(
            run_id=label,
            approach="traditional_ml",
            model_name="log_reg",
            hyperparameters={"class_weight": "balanced", "features": cols},
            split_cfg=split_cfg,
            summary=summary,
            observation=obs,
            stage=stage,
        )
        print(label, summary["macro_f1"])

    # --- Traditional baselines + tuning ---
    pipelines = build_pipelines(feature_cols)
    pipelines.pop("linear_svm", None)
    fitted = {}
    rows = []

    default_hparams = {
        "log_reg": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
        "random_forest": {
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "max_depth": None,
        },
        "grad_boost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    }

    for name, pipe in pipelines.items():
        print(f"\n=== Baseline {name} ===")
        pipe.fit(X_train_f, y_train)
        pred = pipe.predict(X_test_f)
        proba = pipe.predict_proba(X_test_f)
        summary = summarize_classification(y_test, pred, proba, title=f"{name} (baseline)")
        rows.append(summary)
        fitted[name] = pipe
        _log_experiment(
            run_id=f"{name}_baseline",
            approach="traditional_ml",
            model_name=name,
            hyperparameters=default_hparams[name],
            split_cfg=split_cfg,
            summary=summary,
            observation="Default/baseline settings before randomized search.",
            stage="baseline",
        )

    for name in ["log_reg", "random_forest", "grad_boost"]:
        print(f"\n=== Tuning {name} ===")
        search = tune_model(
            name, build_pipelines(feature_cols)[name], X_tune, y_tune, n_iter=6, cv=3
        )
        best = search.best_estimator_
        pred = best.predict(X_test_f)
        proba = best.predict_proba(X_test_f)
        summary = summarize_classification(y_test, pred, proba, title=f"{name} (tuned)")
        rows.append(summary)
        fitted[f"{name}_tuned"] = best
        _log_experiment(
            run_id=f"{name}_tuned",
            approach="traditional_ml",
            model_name=name,
            hyperparameters=search.best_params_,
            split_cfg=split_cfg,
            summary=summary,
            observation=(
                f"RandomizedSearchCV best CV macro-F1={search.best_score_:.4f}; "
                "selected on held-out test macro-F1 among traditional models."
            ),
            stage="tuned",
        )

    best_trad_key, best_trad_f1 = max(
        (
            (k, f1_score(y_test, m.predict(X_test_f), average="macro", zero_division=0))
            for k, m in fitted.items()
        ),
        key=lambda t: t[1],
    )
    best_trad = fitted[best_trad_key]
    print(f"\nBest traditional: {best_trad_key} | macro-F1={best_trad_f1:.4f}")
    trad_pred = best_trad.predict(X_test_f)
    trad_proba = best_trad.predict_proba(X_test_f)
    plot_confusion(y_test, trad_pred, f"Confusion — {best_trad_key}", "cm_best_traditional.png")
    plot_roc_ovr(y_test, trad_proba, f"ROC OvR — {best_trad_key}", "roc_best_traditional.png")
    plt.close("all")

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

    # Error analysis dump
    err = X_test_f.copy()
    err["y_true"] = y_test.to_numpy()
    err["y_pred_trad"] = trad_pred
    err["correct_trad"] = err["y_true"] == err["y_pred_trad"]
    err_by_class = err.groupby("y_true").agg(
        error_rate=("correct_trad", lambda s: float(1 - s.mean())),
        mean_metabolic_burden=("MetabolicBurden", "mean"),
        mean_bmi=("BMI", "mean"),
    )
    err_by_class.to_csv(results_dir / "error_analysis_by_class.csv")

    # --- Deep learning ---
    set_tf_seed(RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_val_s = scaler.transform(X_val_f)
    X_test_s = scaler.transform(X_test_f)
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

        run = init_run(
            name=cfg["name"],
            config={
                "approach": "deep_learning",
                "model_name": cfg["name"],
                "stage": "architecture_search",
                "hyperparameters": cfg,
                "class_weight": class_weight,
                **split_cfg,
            },
            tags=["deep_learning", "architecture_search"],
        )
        for epoch, (tr_loss, va_loss, tr_acc, va_acc) in enumerate(
            zip(
                history.history["loss"],
                history.history["val_loss"],
                history.history["accuracy"],
                history.history["val_accuracy"],
            )
        ):
            log_metrics(
                {
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "train_accuracy": tr_acc,
                    "val_accuracy": va_acc,
                },
                step=epoch,
            )
        log_metrics({k: summary[k] for k in summary if k != "report"})
        finish_run()

        append_experiment_row(
            {
                "run_id": cfg["name"],
                "stage": "architecture_search",
                "approach": "deep_learning",
                "model": cfg["name"],
                "hyperparameters": json.dumps(cfg),
                "dataset_split": split_cfg["split"],
                "n_train": split_cfg["n_train"],
                "n_val": split_cfg["n_val"],
                "n_test": split_cfg["n_test"],
                "accuracy": summary["accuracy"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "roc_auc_ovr_macro": summary["roc_auc_ovr_macro"],
                "observation": (
                    "MLP capacity/dropout/LR variant with class weights + early stopping; "
                    f"best val_loss epoch restored."
                ),
                "wandb_mode": "offline",
            }
        )

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
    print("\nExperiment table:", EXPERIMENT_TABLE_PATH)

    trad_path = save_sklearn_model(best_trad, "best_traditional_ml.joblib")
    joblib.dump(scaler, ROOT / "models" / "dl_standard_scaler.joblib")
    dl_path = save_keras_model(best_dl, "best_deep_learning.keras")

    meta = {
        "raw_rows": int(len(raw)),
        "women_rows": int(len(women_clean)),
        "cleaning_report": clean_report,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "best_traditional": best_trad_key,
        "best_traditional_macro_f1": float(best_trad_f1),
        "best_deep_learning": best_dl_name,
        "best_deep_learning_macro_f1": float(
            f1_score(y_test, dl_pred, average="macro", zero_division=0)
        ),
        "model_paths": {"traditional": str(trad_path), "deep_learning": str(dl_path)},
        "wandb_mode": "offline",
        "wandb_dir": str(ROOT / "wandb"),
        "experiment_table": str(EXPERIMENT_TABLE_PATH),
    }
    (results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\nSaved models:", trad_path, dl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
