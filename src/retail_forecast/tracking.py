"""Optional MLflow integration kept out of the default lightweight path."""

from __future__ import annotations

from pathlib import Path


def log_training_run(
    metrics: dict[str, dict[str, float]],
    artifact_path: str | Path,
    tracking_uri: str | None = None,
) -> None:
    """Log aggregate metrics and the trained bundle when MLflow is installed."""
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional dependency branch
        raise RuntimeError("MLflow is optional. Install with: pip install -e '.[mlflow]'") from error
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    flattened = {f"{partition}_{name}": value for partition, values in metrics.items() for name, value in values.items()}
    with mlflow.start_run():
        mlflow.log_params({"model": "XGBoost", "feature_count": 18, "split_strategy": "chronological"})
        mlflow.log_metrics(flattened)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
