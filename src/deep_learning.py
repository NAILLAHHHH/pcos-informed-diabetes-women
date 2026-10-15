"""TensorFlow / Keras MLP models and tf.data pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import MODELS_DIR, RANDOM_STATE


def set_tf_seed(seed: int = RANDOM_STATE) -> None:
    tf.keras.utils.set_random_seed(seed)


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    seed: int = RANDOM_STATE,
) -> tf.data.Dataset:
    """Build a performant tf.data pipeline for tabular classification."""
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y.astype("int32")))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_mlp(
    n_features: int,
    n_classes: int = 3,
    hidden_units: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.3,
    l2: float = 1e-4,
) -> keras.Model:
    """Functional API MLP suitable for scaled tabular inputs."""
    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2),
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="diabetes_class")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="diabetes_mlp")


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    class_weight: dict | None = None,
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # class_weight is applied at fit-time; kept here for API symmetry
    model._default_class_weight = class_weight  # noqa: SLF001
    return model


def default_callbacks(patience: int = 8) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-6,
        ),
    ]


def save_keras_model(model: keras.Model, filename: str = "best_deep_learning.keras") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    model.save(path)
    return path


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    n_classes = len(classes)
    return {int(c): float(total / (n_classes * cnt)) for c, cnt in zip(classes, counts)}
