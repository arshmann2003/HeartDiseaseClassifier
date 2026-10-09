"""Model training, threshold selection and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET

DATA_FILE = "heart_disease_health_indicators_BRFSS2015.csv"
KAGGLE_DATASET = "alexteboul/heart-disease-health-indicators-dataset"


def load_data(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, downloading it from Kaggle if it is not present locally."""
    path = Path(data_dir) / DATA_FILE
    if not path.exists():
        path = _download(Path(data_dir))
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].astype(int)
    y = df[TARGET].astype(int)
    return X, y


def _download(data_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            f"{data_dir / DATA_FILE} not found. Install `kagglehub` to download it "
            f"automatically, or download it manually from "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
        ) from exc
    import shutil

    source = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / DATA_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(source, data_dir / DATA_FILE))


def candidate_models(random_state: int = 42) -> dict:
    """Models compared in the pipeline.

    None of them use class weighting: weighting distorts predicted probabilities,
    and the app reports probabilities to users. Class imbalance is handled by the
    decision threshold instead.
    """
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=25,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=random_state,
        ),
    }


def ranking_metrics(y_true, y_proba) -> dict:
    """Threshold-independent metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def threshold_metrics(y_true, y_proba, threshold: float) -> dict:
    """Metrics for hard predictions at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "flagged_rate": float(y_pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_recall(y_true, y_proba, target_recall: float) -> float:
    """Highest threshold whose recall is at least `target_recall`.

    The highest such threshold flags the fewest people (best precision) while
    still catching the target share of true cases.
    """
    _, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # recall has one more entry than thresholds; recall[i] pairs with thresholds[i]
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return float(thresholds[0])
    return float(thresholds[valid[-1]])


def explain_prediction(model, row: pd.DataFrame, reference: dict[str, float]) -> pd.DataFrame:
    """Per-feature contribution for a single profile.

    Each feature is swapped, one at a time, for its value in a typical
    reference profile, and the change in predicted risk is recorded. A positive
    contribution means this person's value raises their risk relative to typical.
    """
    base = float(model.predict_proba(row[FEATURE_COLUMNS])[0, 1])
    variants = pd.concat([row[FEATURE_COLUMNS]] * len(FEATURE_COLUMNS), ignore_index=True)
    for i, name in enumerate(FEATURE_COLUMNS):
        variants.loc[i, name] = reference[name]
    swapped = model.predict_proba(variants)[:, 1]
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "contribution": base - swapped})
        .assign(magnitude=lambda d: d["contribution"].abs())
        .sort_values("magnitude", ascending=False)
        .drop(columns="magnitude")
        .reset_index(drop=True)
    )
