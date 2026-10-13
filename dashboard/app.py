"""Small Streamlit interface for inspecting metrics and requesting forecasts."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from retail_forecast.model import load_bundle
from retail_forecast.predict import ForecastRequest, recursive_forecast

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📦", layout="wide")
st.title("Retail Demand Forecasting Platform")
st.caption("Leakage-safe XGBoost forecasts with calendar and planned-promotion features.")

artifact_path = Path(os.getenv("RETAIL_FORECAST_MODEL", "artifacts/model_bundle.joblib"))
if not artifact_path.exists():
    st.error("No model artifact found. Run `python -m retail_forecast.train` first.")
    st.stop()

@st.cache_resource
def get_bundle(path: str):
    return load_bundle(path)

bundle = get_bundle(str(artifact_path))
left, middle, right = st.columns(3)
for column, (label, value) in zip(
    (left, middle, right),
    (("Test WAPE", bundle.metrics["test_xgboost"]["wape"]), ("Test MAE", bundle.metrics["test_xgboost"]["mae"]), ("Test bias", bundle.metrics["test_xgboost"]["bias"])),
):
    column.metric(label, f"{value:.2%}" if label != "Test MAE" else f"{value:.2f}")

catalog = bundle.history[["store_id", "product_id", "category"]].drop_duplicates().sort_values(["store_id", "product_id"])
store_id = st.selectbox("Store", sorted(catalog["store_id"].unique()))
products = catalog[catalog["store_id"] == store_id]["product_id"].tolist()
product_id = st.selectbox("Product", products)
horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=26, value=4)
planned_promotions = st.multiselect("Planned promotion weeks", options=list(range(1, horizon + 1)))

forecast = recursive_forecast(bundle, ForecastRequest(store_id, product_id, horizon, tuple(planned_promotions)))
st.subheader("Demand forecast")
st.line_chart(forecast.set_index("date")["predicted_sales"])
st.dataframe(forecast, use_container_width=True, hide_index=True)

st.subheader("Walk-forward backtest")
st.dataframe(bundle.backtest, use_container_width=True, hide_index=True)
st.caption(f"Data source: {bundle.data_source}")
