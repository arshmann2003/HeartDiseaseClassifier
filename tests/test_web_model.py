import json
from pathlib import Path

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from heart_risk.features import FEATURE_COLUMNS, FEATURES
from heart_risk.web_model import PRESETS, score_logistic

ROOT = Path(__file__).parent.parent
WEB_MODEL = ROOT / "web" / "model.json"


def test_score_matches_sklearn_pipeline():
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=2, n_redundant=0, random_state=0
    )
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    pipeline.fit(X, y)
    columns = ["a", "b", "c", "d"]
    artifact = {
        "feature_columns": columns,
        "mean": pipeline.named_steps["standardscaler"].mean_.tolist(),
        "scale": pipeline.named_steps["standardscaler"].scale_.tolist(),
        "coefficients": pipeline.named_steps["logisticregression"].coef_.ravel().tolist(),
        "intercept": float(pipeline.named_steps["logisticregression"].intercept_[0]),
    }
    row = dict(zip(columns, [0.2, -1.0, 0.5, 2.0], strict=True))
    expected = pipeline.predict_proba([[0.2, -1.0, 0.5, 2.0]])[0, 1]
    assert score_logistic(artifact, row) == pytest.approx(expected, abs=1e-8)


def test_presets_name_every_feature_and_set_age():
    for preset in PRESETS:
        assert set(preset["values"]) == set(FEATURE_COLUMNS)
    by_name = {preset["name"]: preset["values"]["Age"] for preset in PRESETS}
    assert by_name["Healthy 35-year-old"] == 4
    assert by_name["55-year-old with high BP"] == 8
    assert by_name["High-risk 70-year-old"] == 11


def test_published_web_model_scores_presets_in_order():
    artifact = json.loads(WEB_MODEL.read_text())
    assert artifact["scored_model"] == "Logistic Regression"
    assert artifact["feature_columns"] == FEATURE_COLUMNS
    assert len(artifact["coefficients"]) == len(FEATURE_COLUMNS)
    assert [feature["name"] for feature in artifact["features"]] == [feature.name for feature in FEATURES]

    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    published = metadata["results"]["Logistic Regression"]["test_at_threshold"]["threshold"]
    assert artifact["threshold"] == pytest.approx(published, abs=1e-6)

    healthy = next(preset for preset in artifact["presets"] if preset["name"].startswith("Healthy"))
    high = next(preset for preset in artifact["presets"] if preset["name"].startswith("High-risk"))
    assert score_logistic(artifact, high["values"]) > score_logistic(artifact, healthy["values"])
    risk = score_logistic(artifact, {feature.name: feature.default for feature in FEATURES})
    assert 0.0 < risk < 1.0
