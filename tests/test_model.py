import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from heart_risk.features import FEATURE_COLUMNS, FEATURES
from heart_risk.modeling import explain_prediction, threshold_for_recall, threshold_metrics

MODELS = Path(__file__).parent.parent / "models"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS / "heart_risk_model.joblib")


@pytest.fixture(scope="module")
def metadata():
    return json.loads((MODELS / "metadata.json").read_text())


def profile(**overrides) -> pd.DataFrame:
    values = {f.name: f.default for f in FEATURES}
    values.update(overrides)
    return pd.DataFrame([values])[FEATURE_COLUMNS]


def test_schema_matches_trained_model(metadata):
    assert metadata["feature_columns"] == FEATURE_COLUMNS


def test_defaults_are_valid_values():
    for f in FEATURES:
        if f.options:
            assert f.default in f.options, f.name
        else:
            assert f.min_value <= f.default <= f.max_value, f.name


def test_predictions_are_probabilities(model):
    proba = model.predict_proba(profile())[:, 1]
    assert 0.0 <= proba[0] <= 1.0


def test_known_risk_factors_increase_risk(model):
    healthy = profile(Age=4, HighBP=0, HighChol=0, Stroke=0, GenHlth=1)
    at_risk = profile(Age=11, HighBP=1, HighChol=1, Stroke=1, GenHlth=5)
    assert model.predict_proba(at_risk)[0, 1] > 5 * model.predict_proba(healthy)[0, 1]


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.random(5_000) < 0.1
    scores = np.clip(y * 0.3 + rng.random(5_000) * 0.7, 0, 1)
    threshold = threshold_for_recall(y, scores, 0.8)
    assert threshold_metrics(y, scores, threshold)["recall"] >= 0.8


def test_threshold_metrics_counts():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = threshold_metrics(y, proba, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["recall"] == m["precision"] == 0.5


def test_explanation_is_zero_for_reference_profile(model, metadata):
    reference = pd.DataFrame([metadata["reference_profile"]])[FEATURE_COLUMNS]
    contributions = explain_prediction(model, reference, metadata["reference_profile"])
    assert set(contributions["feature"]) == set(FEATURE_COLUMNS)
    assert np.allclose(contributions["contribution"], 0)
