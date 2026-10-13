"""Time-safe feature construction and chronological split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .data import validate_retail_data

GROUP_COLUMNS = ["store_id", "product_id"]
ENCODED_COLUMNS = ["store_id", "product_id", "category", "region"]
FEATURE_COLUMNS = [
    "store_id_code",
    "product_id_code",
    "category_code",
    "region_code",
    "price",
    "promotion",
    "holiday",
    "week_sin",
    "week_cos",
    "month",
    "quarter",
    "trend_week",
    "sales_lag_1",
    "sales_lag_4",
    "sales_lag_13",
    "sales_rolling_mean_4",
    "sales_rolling_mean_13",
    "sales_rolling_std_4",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata required to create production features consistently."""

    category_maps: dict[str, dict[str, int]]
    anchor_date: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FeatureSpec":
        return cls(
            category_maps={
                column: {value: index for index, value in enumerate(sorted(frame[column].astype(str).unique()))}
                for column in ENCODED_COLUMNS
            },
            anchor_date=pd.Timestamp(frame["date"].min()).isoformat(),
        )


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


def build_features(frame: pd.DataFrame, spec: FeatureSpec | None = None, drop_unavailable: bool = True) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build features using only values available strictly before each target week.

    All lags and rolling values use ``shift(1)`` before any aggregation. This is
    intentional: the target week's observed sales can never enter its own features.
    """
    validated = validate_retail_data(frame.dropna(subset=["sales"])) if frame["sales"].notna().all() else frame.copy()
    data = validated.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(GROUP_COLUMNS + ["date"]).reset_index(drop=True)
    actual_spec = spec or FeatureSpec.from_frame(data)
    for column in ENCODED_COLUMNS:
        mapping: Mapping[str, int] = actual_spec.category_maps[column]
        data[f"{column}_code"] = data[column].astype(str).map(mapping).fillna(-1).astype(int)

    iso_week = data["date"].dt.isocalendar().week.astype(int)
    data["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0)
    data["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0)
    data["month"] = data["date"].dt.month.astype(int)
    data["quarter"] = data["date"].dt.quarter.astype(int)
    anchor = pd.Timestamp(actual_spec.anchor_date)
    data["trend_week"] = ((data["date"] - anchor).dt.days // 7).astype(int)

    grouped_sales = data.groupby(GROUP_COLUMNS, sort=False)["sales"]
    for lag in (1, 4, 13):
        data[f"sales_lag_{lag}"] = grouped_sales.shift(lag)
    for window in (4, 13):
        data[f"sales_rolling_mean_{window}"] = grouped_sales.transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=window).mean()
        )
    data["sales_rolling_std_4"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(window=4, min_periods=4).std(ddof=0)
    )
    if drop_unavailable:
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return data, actual_spec


def chronological_split(features: pd.DataFrame, validation_periods: int = 16, test_periods: int = 16) -> TimeSplit:
    """Split every series by shared calendar cutoffs, never by random rows."""
    dates = np.sort(pd.to_datetime(features["date"].unique()))
    if len(dates) <= validation_periods + test_periods + 13:
        raise ValueError("Not enough periods for chronological train/validation/test splits")
    validation_start = pd.Timestamp(dates[-(validation_periods + test_periods)])
    test_start = pd.Timestamp(dates[-test_periods])
    train = features[features["date"] < validation_start].copy()
    validation = features[(features["date"] >= validation_start) & (features["date"] < test_start)].copy()
    test = features[features["date"] >= test_start].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    return TimeSplit(train, validation, test, validation_start, test_start)
