# Retail Demand Forecasting Platform

An independently implemented, portfolio-ready demand forecasting service for
weekly store-product sales. It trains an XGBoost model on leakage-safe lag,
calendar, price, promotion, and holiday features; evaluates a transparent
four-week-lag baseline; serves recursive forecasts through FastAPI; and offers
a compact Streamlit dashboard.

The default run uses a deterministic synthetic retail panel, so the project is
fully reproducible without accepting a data license or downloading a dataset.
See [Data and provenance](#data-and-provenance) and [NOTICE.md](NOTICE.md).

## Highlights

- Chronological train/validation/test windows shared by every store-product
  series; no random splitting.
- Target-history features built with `shift(1)`, so a week's sales never enter
  that same week's feature vector.
- Known-at-forecast-time signals: planned promotions, fixed holiday weeks,
  price, seasonal Fourier terms, and trend.
- Baseline versus XGBoost comparison using MAE, WAPE, and signed bias.
- Three-fold walk-forward backtest that reserves the final test period.
- FastAPI health, catalogue, metrics, and multi-week forecast endpoints.
- Streamlit dashboard, Docker/Compose, GitHub Actions CI, optional MLflow.

## Included results and evaluation scope

The following values are recorded in the supplied training report dated
2026-10-12; they are historical results, not a new execution in this workspace. The deterministic panel contains **4,992
rows** (4 stores × 8 products × 156 weeks). The final 16 shared weeks are the
unseen test period, beginning **2024-09-09**.

| Held-out model | MAE | WAPE | Bias |
| --- | ---: | ---: | ---: |
| Four-week-lag baseline | 15.587 | 12.62% | -5.75% |
| XGBoost | 6.830 | 5.53% | -1.42% |

The supplied validation WAPE was 5.88% (MAE 6.164, bias +0.46%).
These scores evaluate rolling one-week predictions with observed prior-week
sales available. They do **not** measure fixed-origin recursive multi-week
forecast performance. All results use synthetic data and do not establish
accuracy on real retail demand.

Publication review found an off-by-one boundary in the original backtest:
its last fold included the first test week. That boundary is now corrected,
and a regression test checks that every fold ends before the holdout. The
original backtest scores are therefore not presented as valid holdout-isolated
results. Regenerate reports with the corrected implementation before citing
new backtest scores. XGBoost settings are fixed; no hyperparameter search or
model selection is implemented.

See [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) for the supplied test record
and [docs/PUBLICATION_REVIEW.md](docs/PUBLICATION_REVIEW.md) for current checks.

## Current publication verification (2026-10-13)

The corrected implementation passed **9 tests in 2.15s** on Python 3.12.14.
A fresh CLI training run and HTTP API smoke checks also passed.
These freshly generated synthetic-data scores use the rolling one-week
evaluation described above:

| Model | MAE | WAPE | Bias |
| --- | ---: | ---: | ---: |
| baseline | 15.587 | 12.62% | -5.75% |
| xgboost | 6.763 | 5.47% | -1.46% |

Corrected backtest WAPE: 5.32%, 5.88%, 5.81%.
See `docs/evidence/current-metrics.json` for exact values and fold dates.
Docker, Streamlit, MLflow, and hosted CI remain unverified in this workspace.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,dashboard]"
python -m retail_forecast.train
pytest -vv
uvicorn retail_forecast.api:app --reload
```

Open interactive API documentation at `http://127.0.0.1:8000/docs`.

In a second terminal, run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Forecast API

`POST /forecast` accepts a store, product, forecast horizon, and one-based
weeks for planned promotions. Each later forecast is recursively added to the
available demand history, so its lag features contain only observed sales and
earlier predictions.

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "content-type: application/json" \
  -d '{"store_id":"store_01","product_id":"product_001","horizon_weeks":4,"promotion_weeks":[2]}'
```

Other routes:

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Checks whether a trained artifact is present. |
| GET | `/catalog` | Returns valid store-product series. |
| GET | `/model/metrics` | Returns stored test metrics and backtests. |
| POST | `/forecast` | Returns future weekly demand forecasts. |

## Data and provenance

The default data source is **not** a copied dataset. `generate_synthetic_retail_data`
creates a deterministic weekly panel controlled by a seed, with category/store
effects, yearly seasonality, trend, promotion uplift, holidays, price discounts,
noise, and mild autocorrelation. Re-running with the default seed yields the
same 4,992 rows.

To substitute an accessible real retail dataset, provide a CSV with these
columns and pass it to `--data-path`:

```text
date, store_id, product_id, category, region, price, promotion, holiday, sales
```

Dates must be weekly observations, `sales` must be non-negative, and the
`date/store_id/product_id` key must be unique. Preserve the original dataset's
license and attribution when using it; this project does not bundle third-party
retail data.

The user-requested reference repository was available publicly when reviewed on
2026-10-12 and stated it used the MIT License. This project contains a newly
written implementation rather than reused source, data, artifacts, or reported
results. Required reference attribution is preserved in [NOTICE.md](NOTICE.md).

## Evaluation design

1. Features are made from chronological group histories. Every lag and rolling
   statistic begins with `shift(1)`.
2. The last 16 weeks form an untouched test set; the preceding 16 form a
   validation set. All series use the same boundaries.
3. The baseline is `sales_lag_4`. XGBoost uses fixed settings and is evaluated on validation data,
   then refit on train + validation data before the test evaluation. A separate serving model is then fitted on all
   observed history; the stored test scores belong to the evaluation model.
4. Three eight-week walk-forward folds occur before the final held-out period.
5. WAPE is `sum(abs(error)) / sum(abs(actual))`; bias is signed error divided by
   total actual demand. They are aggregate-safe even when individual weeks are
   zero.

This design does not make causal claims about promotions: in production,
promotion and price inputs must be the plans available before the forecast date.

## Training and experiment tracking

```bash
# Regenerate a model and deterministic fallback data if the path is absent.
python -m retail_forecast.train --data-path data/raw/retail_weekly.csv --output-dir artifacts

# Optional MLflow integration.
pip install -e ".[mlflow]"
python -m retail_forecast.train --mlflow --tracking-uri http://127.0.0.1:5000
```

MLflow is optional so the standard install remains lean. When enabled, the
pipeline logs aggregate metrics, model settings, and the trained bundle.

## Containers and CI

```bash
docker compose up --build
```

The API is exposed at port 8000 and the dashboard at 8501. The Docker build
trains the deterministic fallback model during image construction. GitHub
Actions runs the test suite on Python 3.10 and 3.12 and performs a fresh CLI
training smoke test on pushes and pull requests.

## Repository layout

```text
src/retail_forecast/   data contract, features, model, evaluation, API, training
dashboard/app.py       Streamlit interface
tests/                 data, leakage, metrics, pipeline, and API behavior tests
docs/                  architecture and verified test record
.github/workflows/     continuous-integration workflow
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design and operational
notes.
