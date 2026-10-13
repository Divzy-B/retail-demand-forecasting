from __future__ import annotations

from pathlib import Path

import pytest

from retail_forecast.data import SyntheticDataConfig, generate_synthetic_retail_data
from retail_forecast.train import train_pipeline


@pytest.fixture(scope="session")
def small_data():
    return generate_synthetic_retail_data(SyntheticDataConfig(weeks=64, stores=2, products=3, seed=11))


@pytest.fixture(scope="session")
def trained_bundle(small_data):
    # Keep test artifacts within the checked-out workspace so test execution does
    # not depend on an operating system's temporary-directory permissions.
    output = Path(__file__).parents[1] / "test_runtime_artifacts"
    output.mkdir(exist_ok=True)
    return train_pipeline(small_data, output, data_source="test_synthetic")
