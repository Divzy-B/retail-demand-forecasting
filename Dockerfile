FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RETAIL_FORECAST_MODEL=/app/artifacts/model_bundle.joblib

COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"
COPY dashboard ./dashboard
COPY data ./data
RUN python -m retail_forecast.train --output-dir /app/artifacts --data-path /app/data/raw/retail_weekly.csv

EXPOSE 8000
CMD ["uvicorn", "retail_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
