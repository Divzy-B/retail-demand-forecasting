"""XGBoost model helpers and portable trained-bundle persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FeatureSpec


@dataclass
class ForecastBundle:
    """Everything the prediction service needs, persisted as one artifact."""

    model: Any
    feature_columns: list[str]
    feature_spec: FeatureSpec
    history: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    backtest: pd.DataFrame
    data_source: str


def fit_xgboost(train: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> Any:
    """Fit the production XGBoost regressor with deterministic, CPU-safe settings."""
    try:
        from xgboost import XGBRegressor
    except ImportError as error:  # pragma: no cover - dependency contract error
        raise RuntimeError("XGBoost is required. Install the project dependencies first.") from error

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train[feature_columns], train["sales"])
    return model


def predict_nonnegative(model: Any, features: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    return np.maximum(0.0, model.predict(features[feature_columns]))


def save_bundle(bundle: ForecastBundle, path: str | Path) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return artifact_path


def load_bundle(path: str | Path) -> ForecastBundle:
    return joblib.load(Path(path))
