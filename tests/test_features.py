import numpy as np

from retail_forecast.features import FEATURE_COLUMNS, build_features, chronological_split


def test_features_do_not_use_current_or_future_target(small_data):
    original, _ = build_features(small_data)
    changed = small_data.copy()
    changed.loc[changed.index[-1], "sales"] = 999999.0
    altered, _ = build_features(changed)
    before_last = original[original["date"] < original["date"].max()].sort_values(["store_id", "product_id", "date"])
    altered_before_last = altered[altered["date"] < altered["date"].max()].sort_values(["store_id", "product_id", "date"])
    assert np.allclose(before_last[FEATURE_COLUMNS], altered_before_last[FEATURE_COLUMNS])
    original_last = original.iloc[-1][FEATURE_COLUMNS].to_numpy(dtype=float)
    altered_last = altered.iloc[-1][FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(original_last, altered_last)


def test_chronological_split_is_strictly_ordered(small_data):
    features, _ = build_features(small_data)
    split = chronological_split(features, validation_periods=10, test_periods=10)
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
