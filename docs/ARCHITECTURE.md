# Architecture

```text
CSV or deterministic generator
             |
        validate data
             |
  leakage-safe feature builder
             |
 chronology split + walk-forward backtest
             |
 XGBoost model / lag-4 baseline ---> metrics.json and backtest.csv
             |
      model_bundle.joblib
       /              \
 FastAPI service    Streamlit dashboard
```

## Artifact contract

`model_bundle.joblib` contains the fitted XGBoost model, exact feature list,
categorical mappings, raw historical panel used for recursive lags, stored
metrics, walk-forward results, and source label. This keeps the API and
dashboard consistent with training.

## Operational notes

- Forecast only store-product pairs present in model history. Retrain to add a
  new store or product.
- `promotion_weeks` means known promotion plans. Omit unplanned promotions
  rather than estimating their effect at serving time.
- Treat error changes, prolonged positive bias, and data-contract violations as
  retraining or investigation triggers.
- Model artifacts are generated and excluded from Git. Use a model registry or
  object store with versioning for production deployment.
- Test and validation bounds must move forward with retraining. Do not shuffle
  weekly records or calculate a rolling target statistic without first shifting.
