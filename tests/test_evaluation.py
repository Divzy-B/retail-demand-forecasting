import numpy as np

from retail_forecast.evaluation import forecast_metrics


def test_metrics_match_hand_calculation():
    metrics = forecast_metrics(np.array([10.0, 20.0]), np.array([8.0, 25.0]))
    assert metrics["mae"] == 3.5
    assert metrics["wape"] == 7 / 30
    assert metrics["bias"] == 3 / 30


def test_walk_forward_never_enters_final_holdout(small_data):
    import pandas as pd
    from retail_forecast.evaluation import walk_forward_backtest
    from retail_forecast.features import build_features

    featured, spec = build_features(small_data)
    dates = sorted(featured["date"].unique())
    cutoff = pd.Timestamp(dates[-16])
    observed = []

    def predict(train, validation):
        assert train["date"].max() < validation["date"].min()
        assert validation["date"].max() < cutoff
        assert validation["date"].nunique() == 8
        observed.append(validation["date"].max())
        return validation["sales_lag_4"].to_numpy()

    result = walk_forward_backtest(small_data, spec, predict)
    assert len(result) == 3
    assert len(observed) == 3
