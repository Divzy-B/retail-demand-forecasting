import json


def test_training_writes_artifacts_and_backtest(trained_bundle):
    bundle, artifact = trained_bundle
    assert artifact.exists()
    assert len(bundle.backtest) == 3
    assert {"mae", "wape", "bias"} <= set(bundle.metrics["test_xgboost"])
    report = json.loads(artifact.with_name("metrics.json").read_text())
    assert report["rows"] > 0
    assert len(report["walk_forward"]) == 3
