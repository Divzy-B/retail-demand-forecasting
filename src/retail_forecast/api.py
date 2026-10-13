"""FastAPI service for the saved retail demand forecasting model."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ForecastBundle, load_bundle
from .predict import ForecastRequest, recursive_forecast


class ForecastPayload(BaseModel):
    store_id: str = Field(min_length=1, examples=["store_01"])
    product_id: str = Field(min_length=1, examples=["product_001"])
    horizon_weeks: int = Field(default=4, ge=1, le=26)
    promotion_weeks: list[int] = Field(default_factory=list, description="One-based forecast weeks with a planned promotion")


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float
    promotion: int
    holiday: int


class ForecastResponse(BaseModel):
    store_id: str
    product_id: str
    model: str = "xgboost"
    forecasts: list[ForecastPoint]


def create_app(bundle_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Retail Demand Forecasting API", version="0.1.0")
    app.state.bundle_path = Path(bundle_path or os.getenv("RETAIL_FORECAST_MODEL", "artifacts/model_bundle.joblib"))
    app.state.bundle = None

    def get_bundle() -> ForecastBundle:
        if app.state.bundle is None:
            if not app.state.bundle_path.exists():
                raise HTTPException(status_code=503, detail="Model artifact is unavailable. Run the training command first.")
            app.state.bundle = load_bundle(app.state.bundle_path)
        return app.state.bundle

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_ready": app.state.bundle_path.exists()}

    @app.get("/catalog")
    def catalog() -> dict[str, object]:
        bundle = get_bundle()
        pairs = bundle.history[["store_id", "product_id", "category", "region"]].drop_duplicates()
        return {"series": pairs.to_dict(orient="records")}

    @app.get("/model/metrics")
    def model_metrics() -> dict[str, object]:
        bundle = get_bundle()
        return {"model": "xgboost", "data_source": bundle.data_source, "metrics": bundle.metrics, "walk_forward": bundle.backtest.to_dict(orient="records")}

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastPayload) -> ForecastResponse:
        try:
            result = recursive_forecast(
                get_bundle(),
                ForecastRequest(
                    store_id=payload.store_id,
                    product_id=payload.product_id,
                    horizon_weeks=payload.horizon_weeks,
                    promotion_weeks=tuple(payload.promotion_weeks),
                ),
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        points = [
            ForecastPoint(
                date=row.date.date().isoformat(),
                predicted_sales=round(float(row.predicted_sales), 2),
                promotion=int(row.promotion),
                holiday=int(row.holiday),
            )
            for row in result.itertuples(index=False)
        ]
        return ForecastResponse(store_id=payload.store_id, product_id=payload.product_id, forecasts=points)

    return app


app = create_app()
