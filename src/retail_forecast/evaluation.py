"""Forecast metrics and leakage-safe walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FeatureSpec, build_features


def forecast_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """Return MAE, WAPE and signed bias percentage; safe for zero-demand rows."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(predicted, dtype=float))
    denominator = float(np.abs(y_true).sum())
    errors = y_pred - y_true
    return {
        "mae": float(np.abs(errors).mean()),
        "wape": float(np.abs(errors).sum() / denominator) if denominator else 0.0,
        "bias": float(errors.sum() / denominator) if denominator else 0.0,
    }


def walk_forward_backtest(
    raw_data: pd.DataFrame,
    spec: FeatureSpec,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    folds: int = 3,
    horizon: int = 8,
    holdout_periods: int = 16,
) -> pd.DataFrame:
    """Evaluate sequential forecast origins while reserving the final holdout.

    ``fit_predict`` receives rows at or before an origin and the immediately
    following horizon. It must not inspect later rows.
    """
    featured, _ = build_features(raw_data, spec=spec)
    dates = np.sort(featured["date"].unique())
    last_eligible_origin = len(dates) - holdout_periods - horizon - 1
    origins = [last_eligible_origin - horizon * index for index in range(folds - 1, -1, -1)]
    if min(origins) <= 0:
        raise ValueError("Not enough history for requested walk-forward backtest")
    rows: list[dict[str, object]] = []
    for fold, origin_index in enumerate(origins, start=1):
        origin = pd.Timestamp(dates[origin_index])
        future_end = pd.Timestamp(dates[origin_index + horizon])
        train = featured[featured["date"] <= origin]
        validation = featured[(featured["date"] > origin) & (featured["date"] <= future_end)]
        prediction = fit_predict(train, validation)
        metric = forecast_metrics(validation["sales"], prediction)
        rows.append({"fold": fold, "origin_date": origin.date().isoformat(), "rows": len(validation), **metric})
    return pd.DataFrame(rows)


def baseline_from_lag_4(features: pd.DataFrame) -> np.ndarray:
    """Transparent benchmark: demand observed four weeks earlier."""
    return features["sales_lag_4"].to_numpy(dtype=float)
