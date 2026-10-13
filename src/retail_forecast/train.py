"""Command-line training pipeline with chronological evaluation and backtesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import SyntheticDataConfig, load_or_create_data
from .evaluation import baseline_from_lag_4, forecast_metrics, walk_forward_backtest
from .features import FEATURE_COLUMNS, FeatureSpec, build_features, chronological_split
from .model import ForecastBundle, fit_xgboost, predict_nonnegative, save_bundle
from .tracking import log_training_run


def train_pipeline(
    data: pd.DataFrame,
    output_dir: str | Path,
    data_source: str = "synthetic_fallback",
    enable_mlflow: bool = False,
    tracking_uri: str | None = None,
) -> tuple[ForecastBundle, Path]:
    """Train, evaluate, backtest, and persist a production-ready forecasting bundle."""
    features, spec = build_features(data)
    split = chronological_split(features)
    validation_model = fit_xgboost(split.train)
    validation_metrics = {
        "baseline": forecast_metrics(split.validation["sales"], baseline_from_lag_4(split.validation)),
        "xgboost": forecast_metrics(split.validation["sales"], predict_nonnegative(validation_model, split.validation)),
    }

    train_and_validation = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = fit_xgboost(train_and_validation)
    test_metrics = {
        "baseline": forecast_metrics(split.test["sales"], baseline_from_lag_4(split.test)),
        "xgboost": forecast_metrics(split.test["sales"], predict_nonnegative(final_model, split.test)),
    }

    def fold_predict(train: pd.DataFrame, validation: pd.DataFrame):
        return predict_nonnegative(fit_xgboost(train), validation)

    backtest = walk_forward_backtest(data, spec, fold_predict)
    bundle = ForecastBundle(
        model=fit_xgboost(features),
        feature_columns=FEATURE_COLUMNS,
        feature_spec=spec,
        history=data.copy(),
        metrics={"validation_xgboost": validation_metrics["xgboost"], "test_xgboost": test_metrics["xgboost"], "test_baseline": test_metrics["baseline"]},
        backtest=backtest,
        data_source=data_source,
    )
    output_dir = Path(output_dir)
    artifact_path = save_bundle(bundle, output_dir / "model_bundle.joblib")
    report = {
        "validation": validation_metrics,
        "test": test_metrics,
        "walk_forward": backtest.to_dict(orient="records"),
        "data_source": data_source,
        "rows": int(len(data)),
        "validation_start": split.validation_start.date().isoformat(),
        "test_start": split.test_start.date().isoformat(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    backtest.to_csv(output_dir / "walk_forward_backtest.csv", index=False)
    if enable_mlflow:
        log_training_run(bundle.metrics, artifact_path, tracking_uri)
    return bundle, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Retail Demand Forecasting Platform")
    parser.add_argument("--data-path", default="data/raw/retail_weekly.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true", help="Log the run when optional MLflow is installed")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    data_path = Path(args.data_path)
    existed = data_path.exists()
    data = load_or_create_data(data_path, SyntheticDataConfig(seed=args.seed))
    source = f"csv:{data_path}" if existed else "deterministic_synthetic_fallback"
    bundle, artifact = train_pipeline(data, args.output_dir, source, args.mlflow, args.tracking_uri)
    print(json.dumps({"artifact": str(artifact), "metrics": bundle.metrics, "backtest_folds": len(bundle.backtest)}, indent=2))


if __name__ == "__main__":
    main()
