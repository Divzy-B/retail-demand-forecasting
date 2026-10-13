import pandas as pd
import pytest

from retail_forecast.data import SyntheticDataConfig, generate_synthetic_retail_data, validate_retail_data


def test_generator_is_reproducible_and_complete():
    config = SyntheticDataConfig(weeks=40, stores=2, products=3, seed=7)
    first = generate_synthetic_retail_data(config)
    second = generate_synthetic_retail_data(config)
    assert first.equals(second)
    assert len(first) == 40 * 2 * 3
    assert first["sales"].ge(0).all()
    assert first.groupby(["store_id", "product_id"])["date"].nunique().eq(40).all()


def test_validation_rejects_duplicate_keys():
    data = generate_synthetic_retail_data(SyntheticDataConfig(weeks=32, stores=1, products=1))
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_retail_data(duplicated)
