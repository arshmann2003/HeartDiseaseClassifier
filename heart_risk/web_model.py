"""Serialize logistic regression so a static page can score it.

The public demo used to boot a Streamlit server on Community Cloud. After a
few idle hours that server is gone, and the first person to open the link
waits through a cold start. That wait is the whole visit when the link is on
a resume or a GitHub README.

Logistic regression is the model we can serve without a process. On the
held-out test set it lands within 0.003 ROC-AUC of the gradient boosting
model that won validation PR-AUC, and its parameters are a standard scaler
plus one coefficient per question. The page scores those numbers in the
browser. Gradient boosting stays in the comparison table; it is no longer
required to answer a single profile.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .features import FEATURE_COLUMNS, FEATURES, YES_NO

WEB_MODEL_PATH = Path("web/model.json")
GROUPS = ["Demographics", "Medical history", "Lifestyle", "Health status", "Healthcare access"]
SLIDER_ORDINALS = {"Age", "GenHlth"}
RADIO_ORDINALS = {"Diabetes"}

# Age is set explicitly on every preset. The 55-year-old label is the 55–59
# bucket (code 8); tying it to Feature.default would make the label lie if
# that default ever moved.
PRESETS = [
    {
        "name": "Healthy 35-year-old",
        "values": {
            "Age": 4,
            "Sex": 0,
            "Education": 6,
            "Income": 8,
            "HighBP": 0,
            "HighChol": 0,
            "CholCheck": 1,
            "Diabetes": 0,
            "Stroke": 0,
            "BMI": 23,
            "Smoker": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "GenHlth": 2,
            "PhysHlth": 0,
            "MentHlth": 2,
            "DiffWalk": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
        },
    },
    {
        "name": "55-year-old with high BP",
        "values": {
            "Age": 8,
            "Sex": 1,
            "Education": 5,
            "Income": 6,
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "Diabetes": 0,
            "Stroke": 0,
            "BMI": 29,
            "Smoker": 0,
            "PhysActivity": 0,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "GenHlth": 3,
            "PhysHlth": 0,
            "MentHlth": 0,
            "DiffWalk": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
        },
    },
    {
        "name": "High-risk 70-year-old",
        "values": {
            "Age": 11,
            "Sex": 1,
            "Education": 4,
            "Income": 4,
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "Diabetes": 2,
            "Stroke": 0,
            "BMI": 32,
            "Smoker": 1,
            "PhysActivity": 0,
            "Fruits": 0,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "GenHlth": 4,
            "PhysHlth": 10,
            "MentHlth": 3,
            "DiffWalk": 1,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
        },
    },
]


def input_control(feature) -> str:
    """Which control the static page should render for this question."""
    if feature.kind == "binary" and feature.options == YES_NO:
        return "toggle"
    if feature.name == "BMI":
        return "number"
    if feature.kind == "numeric":
        return "slider"
    if feature.name in SLIDER_ORDINALS:
        return "select-slider"
    if feature.kind == "binary" or feature.name in RADIO_ORDINALS:
        return "radio"
    return "select"


def feature_schema() -> list[dict]:
    rows = []
    for feature in FEATURES:
        rows.append(
            {
                "name": feature.name,
                "label": feature.label,
                "kind": feature.kind,
                "group": feature.group,
                "help": feature.help,
                "control": input_control(feature),
                "options": [{"value": int(key), "label": label} for key, label in feature.options.items()],
                "min_value": feature.min_value,
                "max_value": feature.max_value,
                "default": int(feature.default),
            }
        )
    return rows


def score_logistic(artifact: dict, values: dict) -> float:
    """Probability from a serialized StandardScaler + LogisticRegression.

    This is the contract the browser implements. Keep the two copies the same:
    standardize each answer, take the linear score, then apply the sigmoid.
    """
    logit = float(artifact["intercept"])
    for index, name in enumerate(artifact["feature_columns"]):
        scale = float(artifact["scale"][index])
        centered = float(values[name]) - float(artifact["mean"][index])
        standardized = 0.0 if scale == 0 else centered / scale
        logit += float(artifact["coefficients"][index]) * standardized
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def artifact_from_pipeline(pipeline, metadata: dict) -> dict:
    """Build the JSON document the static page fetches once."""
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    logistic = metadata["results"]["Logistic Regression"]
    columns = list(FEATURE_COLUMNS)
    coefficients = [float(value) for value in classifier.coef_.ravel()]
    if len(coefficients) != len(columns):
        raise ValueError(
            f"logistic regression has {len(coefficients)} coefficients for {len(columns)} features"
        )
    return {
        "scored_model": "Logistic Regression",
        "selection_model": metadata["model_name"],
        "selection_threshold": float(metadata["threshold"]),
        "reason": (
            "The page scores logistic regression in the browser so a visit does not "
            "wait for a Python process to wake up. Gradient boosting won validation "
            "PR-AUC and stays in the comparison; the gap on the test set is small "
            "enough that the simpler model is the one we serve."
        ),
        "feature_columns": columns,
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "coefficients": coefficients,
        "intercept": float(classifier.intercept_.ravel()[0]),
        "threshold": float(logistic["test_at_threshold"]["threshold"]),
        "prevalence": float(metadata["prevalence"]),
        "target_recall": float(metadata["target_recall"]),
        "n_rows": int(metadata["n_rows"]),
        "split_sizes": metadata["split_sizes"],
        "trained_at": metadata.get("trained_at"),
        "sklearn_version": metadata.get("sklearn_version"),
        "reference_profile": {key: float(value) for key, value in metadata["reference_profile"].items()},
        "prevalence_by_group": metadata["prevalence_by_group"],
        "feature_importance": metadata.get("feature_importance", []),
        "results": metadata["results"],
        "features": feature_schema(),
        "groups": GROUPS,
        "presets": PRESETS,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_web_model(pipeline, metadata: dict, path: Path = WEB_MODEL_PATH) -> Path:
    artifact = _jsonable(artifact_from_pipeline(pipeline, metadata))
    if list(artifact["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("web model column order drifted from the feature schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return path


def refresh_logistic_artifact(
    metadata_path: Path = Path("models/metadata.json"),
    path: Path = WEB_MODEL_PATH,
) -> Path:
    """Fit logistic regression on the canonical split and write the web artifact.

    Does not replace models/heart_risk_model.joblib. Gradient boosting remains
    the selected model in metadata.json. The published logistic-regression
    metrics are kept when the refit reproduces them, so the page and the
    README keep describing the same test set.
    """
    import sklearn
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    from .modeling import (
        candidate_models,
        load_data,
        ranking_metrics,
        threshold_for_recall,
        threshold_metrics,
    )

    seed = 42
    target_recall = 0.80
    metadata = json.loads(metadata_path.read_text())
    published = metadata["results"]["Logistic Regression"]

    X, y = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    model = clone(candidate_models(seed)["Logistic Regression"]).fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    threshold = threshold_for_recall(y_val, val_proba, target_recall)
    fresh = {
        "validation": ranking_metrics(y_val, val_proba),
        "test": ranking_metrics(y_test, test_proba),
        "test_at_threshold": threshold_metrics(y_test, test_proba, threshold),
    }
    published_auc = published["test"]["roc_auc"]
    fresh_auc = fresh["test"]["roc_auc"]
    if abs(fresh_auc - published_auc) > 5e-3:
        raise RuntimeError(
            "Refit logistic regression test ROC-AUC "
            f"{fresh_auc:.4f} drifted from the published {published_auc:.4f}. "
            "Refusing to ship weights that do not match the reported evaluation."
        )
    # Prefer the published table when the refit landed on it, so rounding in
    # the README does not fork into a second set of test numbers.
    published_threshold = published["test_at_threshold"]["threshold"]
    same_auc = abs(fresh_auc - published_auc) <= 1e-6
    same_threshold = abs(threshold - published_threshold) <= 1e-6
    if same_auc and same_threshold:
        metadata["results"]["Logistic Regression"] = published
    else:
        metadata["results"]["Logistic Regression"] = fresh
    metadata["sklearn_version"] = sklearn.__version__
    return write_web_model(model, metadata, path)
