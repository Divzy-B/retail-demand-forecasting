# Supplied test record (historical)

Executed 2026-10-12 using Python 3.12.14:

```text
$ pytest -vv
8 passed in 10.35s
```

Covered behavior:

- Deterministic synthetic data generation and data-contract validation.
- Chronological split ordering and leakage protection for current/future target
  values.
- Exact MAE, WAPE, and bias calculations.
- Training artifact, metrics report, and three-fold walk-forward output.
- FastAPI route behavior for health, forecast, promotion plan propagation, and
  unknown-series handling.

The default training command was also executed successfully on the full
4,992-row synthetic panel. Exact validation, test, and walk-forward values are
listed in the repository README and emitted to `artifacts/metrics.json`.

Streamlit 1.65.0 and the FastAPI route table were smoke-checked. A Docker build
was not executed because the local Docker Desktop Linux engine was unavailable
at verification time; the Dockerfile and Compose configuration are included
for a Docker-enabled environment.

## Cloud publication verification — 2026-10-13

Python 3.12.14: `.venv/bin/python -m pytest -vv` completed with
**9 passed in 2.15s**, including the new holdout-boundary regression test.
The original eight tests invoke API endpoint functions directly; a separate
HTTP TestClient smoke check verified health, catalogue, metrics, four-week
forecast, promotion propagation, 404, 422, and missing-model 503 responses.

Fresh default synthetic-data CLI training completed successfully after the
backtest correction. See `evidence/current-pytest.txt`,
`evidence/current-http-smoke.txt`, `evidence/current-training.txt`,
`evidence/current-metrics.json`, and `evidence/current-environment.txt`.

Docker is unavailable in this cloud environment. Docker build, Streamlit UI,
MLflow integration, Python 3.10, and hosted GitHub Actions were not executed
here. Earlier task claims above remain supplied historical evidence.
