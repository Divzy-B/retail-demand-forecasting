"""Recursive multi-week forecasting from the saved, time-safe model bundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_features
from .model import ForecastBundle, predict_nonnegative


@dataclass(frozen=True)
class ForecastRequest:
    store_id: str
    product_id: str
    horizon_weeks: int = 4
    promotion_weeks: tuple[int, ...] = ()


def recursive_forecast(bundle: ForecastBundle, request: ForecastRequest) -> pd.DataFrame:
    """Forecast a store-product series, feeding prior predictions into later lags.

    The caller may mark planned promotions by one-based forecast week. Calendar
    features are deterministic, and all demand features are historical observed or
    previously predicted values, never hidden future sales.
    """
    if not 1 <= request.horizon_weeks <= 26:
        raise ValueError("horizon_weeks must be between 1 and 26")
    history = bundle.history.copy()
    series = history[(history["store_id"] == request.store_id) & (history["product_id"] == request.product_id)]
    if series.empty:
        raise KeyError("Unknown store_id/product_id combination")
    series = series.sort_values("date").copy()
    latest_date = pd.Timestamp(series["date"].max())
    template = series.iloc[-1].to_dict()
    future_rows: list[dict[str, object]] = []
    predictions: list[float] = []
    for week in range(1, request.horizon_weeks + 1):
        future_date = latest_date + pd.Timedelta(days=7 * week)
        record = template.copy()
        record["date"] = future_date
        record["promotion"] = int(week in request.promotion_weeks)
        record["holiday"] = int(_holiday_week(future_date))
        record["price"] = float(template["price"]) * (0.88 if record["promotion"] else 1.0)
        record["sales"] = np.nan
        candidate = pd.concat([series, pd.DataFrame(future_rows + [record])], ignore_index=True)
        candidate_features, _ = build_features(candidate, spec=bundle.feature_spec, drop_unavailable=False)
        last = candidate_features.iloc[[-1]]
        prediction = float(predict_nonnegative(bundle.model, last, bundle.feature_columns)[0])
        record["sales"] = prediction
        future_rows.append(record)
        predictions.append(prediction)
    output = pd.DataFrame(future_rows)
    return output[["date", "store_id", "product_id", "promotion", "holiday", "sales"]].rename(columns={"sales": "predicted_sales"})


def _holiday_week(date: pd.Timestamp) -> int:
    end = date + pd.Timedelta(days=6)
    dates = pd.date_range(date, end, freq="D")
    return int(any((day.month == 12 and day.day in {24, 25, 26}) or (day.month == 11 and 22 <= day.day <= 28) for day in dates))
