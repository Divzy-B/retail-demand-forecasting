"""Data contracts and a deterministic synthetic retail panel generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "store_id",
    "product_id",
    "category",
    "region",
    "price",
    "promotion",
    "holiday",
    "sales",
}


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for a reproducible weekly store-product panel."""

    start_date: str = "2022-01-03"
    weeks: int = 156
    stores: int = 4
    products: int = 8
    seed: int = 42


def _holiday_week(date: pd.Timestamp) -> int:
    """Known calendar signal: weeks containing US year-end retail holidays."""
    end = date + pd.Timedelta(days=6)
    dates = pd.date_range(date, end, freq="D")
    return int(any((day.month == 12 and day.day in {24, 25, 26}) or (day.month == 11 and 22 <= day.day <= 28) for day in dates))


def generate_synthetic_retail_data(config: SyntheticDataConfig = SyntheticDataConfig()) -> pd.DataFrame:
    """Create a retail-like panel without downloading or redistributing third-party data.

    Promotions and holidays are features known when a forecast is made; sales noise is
    generated independently. The function is deterministic for a supplied seed.
    """
    if config.weeks < 32:
        raise ValueError("weeks must be at least 32 to support time-series features")
    if config.stores < 1 or config.products < 1:
        raise ValueError("stores and products must be positive")

    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start_date, periods=config.weeks, freq="W-MON")
    categories = ("grocery", "household", "personal_care", "seasonal")
    regions = ("north", "south", "east", "west")
    product_base = rng.uniform(35, 115, size=config.products)
    product_price = rng.uniform(3.5, 30, size=config.products)
    store_multiplier = rng.uniform(0.75, 1.35, size=config.stores)
    rows: list[dict[str, object]] = []

    for store_idx in range(config.stores):
        for product_idx in range(config.products):
            previous = product_base[product_idx] * store_multiplier[store_idx]
            category = categories[product_idx % len(categories)]
            for week_idx, date in enumerate(dates):
                holiday = _holiday_week(date)
                promotion = int(rng.random() < (0.22 if holiday else 0.12))
                annual = 1.0 + 0.16 * np.sin(2 * np.pi * week_idx / 52 + product_idx / 3)
                trend = 1.0 + 0.0015 * week_idx
                promo_lift = 1.0 + 0.28 * promotion
                holiday_lift = 1.0 + (0.20 if holiday and category in {"grocery", "seasonal"} else 0.05 * holiday)
                expectation = product_base[product_idx] * store_multiplier[store_idx] * annual * trend * promo_lift * holiday_lift
                # Mild autocorrelation makes lag features useful without exposing the future.
                sales = max(0.0, 0.80 * expectation + 0.20 * previous + rng.normal(0, 7.0))
                previous = sales
                price = product_price[product_idx] * (0.88 if promotion else 1.0)
                rows.append(
                    {
                        "date": date,
                        "store_id": f"store_{store_idx + 1:02d}",
                        "product_id": f"product_{product_idx + 1:03d}",
                        "category": category,
                        "region": regions[store_idx % len(regions)],
                        "price": round(float(price), 2),
                        "promotion": promotion,
                        "holiday": holiday,
                        "sales": round(float(sales), 2),
                    }
                )
    return pd.DataFrame(rows).sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def validate_retail_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the public project data contract."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="raise")
    for column in ("price", "promotion", "holiday", "sales"):
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    if clean[["date", "store_id", "product_id", "sales"]].isnull().any().any():
        raise ValueError("date, store_id, product_id, and sales may not be null")
    if (clean["sales"] < 0).any():
        raise ValueError("sales must be non-negative")
    duplicate_keys = clean.duplicated(["date", "store_id", "product_id"])
    if duplicate_keys.any():
        raise ValueError("date/store_id/product_id combinations must be unique")
    return clean.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def load_or_create_data(path: str | Path, config: SyntheticDataConfig = SyntheticDataConfig()) -> pd.DataFrame:
    """Load a valid CSV, or materialize the deterministic fallback at *path*."""
    path = Path(path)
    if path.exists():
        return validate_retail_data(pd.read_csv(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = generate_synthetic_retail_data(config)
    generated.to_csv(path, index=False)
    return generated
