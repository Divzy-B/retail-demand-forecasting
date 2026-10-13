import pytest
from fastapi import HTTPException

from retail_forecast.api import ForecastPayload, create_app


def endpoint_for(app, path):
    return next(route.endpoint for route in app.routes if route.path == path)


def test_health_and_forecast_endpoint(trained_bundle):
    _, artifact = trained_bundle
    app = create_app(artifact)
    assert endpoint_for(app, "/health")()["model_ready"] is True
    response = endpoint_for(app, "/forecast")(
        ForecastPayload(store_id="store_01", product_id="product_001", horizon_weeks=3, promotion_weeks=[2])
    )
    assert len(response.forecasts) == 3
    assert response.forecasts[1].promotion == 1


def test_unknown_series_returns_not_found(trained_bundle):
    _, artifact = trained_bundle
    forecast = endpoint_for(create_app(artifact), "/forecast")
    with pytest.raises(HTTPException) as error:
        forecast(ForecastPayload(store_id="missing", product_id="product_001"))
    assert error.value.status_code == 404
