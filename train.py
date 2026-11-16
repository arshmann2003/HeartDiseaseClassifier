"""Train, evaluate and export the heart disease risk screening model.

    python train.py                  # full pipeline (includes hyperparameter search)
    python train.py --skip-tuning    # faster, uses default gradient boosting settings
    python train.py --web-only       # refresh web/model.json from logistic regression

heart_risk.cleaning runs first. It drops missing values and values outside
the allowed range, counts duplicate rows, and prints that audit. The split has
not happened yet. Data is then split 60/20/20
into train / validation / test. Model selection,
hyperparameter search and the decision threshold only ever see train and
validation data; the test set is scored once at the end.

The joblib artifact is the model that won validation PR-AUC. web/model.json
is always the logistic regression, because the public demo scores that model
in the browser instead of cold-starting a Python server.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from heart_risk.features import FEATURE_BY_NAME, FEATURE_COLUMNS
from heart_risk.modeling import (
    candidate_models,
    load_data,
    ranking_metrics,
    threshold_for_recall,
    threshold_metrics,
)
from heart_risk.web_model import refresh_logistic_artifact, write_web_model

SEED = 42
TARGET_RECALL = 0.80
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")
COLORS = {"Logistic Regression": "#8c8c8c", "Random Forest": "#4c9be8", "Gradient Boosting": "#e8554e"}
plt = None


def _configure_plots():
    """Import pyplot only for a full training run.

    `--web-only` refits logistic regression and writes JSON. It should not
    require the plotting dependencies.
    """
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def tune_gradient_boosting(model, X_train, y_train):
    from scipy.stats import loguniform, randint

    search = RandomizedSearchCV(
        model,
        param_distributions={
            "learning_rate": loguniform(0.02, 0.2),
            "max_leaf_nodes": randint(15, 64),
            "min_samples_leaf": randint(20, 200),
            "l2_regularization": loguniform(0.01, 10),
        },
        n_iter=20,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  best CV PR-AUC {search.best_score_:.4f} with {search.best_params_}")
    return search.best_estimator_, search.best_params_


class TrainValTest(NamedTuple):
    """The 60/20/20 split. Each later step only receives the part it is allowed to see."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def write_cleaning_records(report) -> None:
    """Save the three cleaning tables before any model is fit."""
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    report.missing.to_csv(reports / "dropped_missing.csv", index=False)
    report.out_of_range.to_csv(reports / "dropped_out_of_range.csv", index=False)
    report.duplicates.to_csv(reports / "duplicate_rows.csv", index=False)
    print(
        "Record tables written to reports/: "
        f"{report.dropped_missing:,} missing, "
        f"{report.dropped_out_of_range:,} out of range, "
        f"{report.duplicate_rows_kept:,} duplicates kept"
    )


def split_train_val_test(X: pd.DataFrame, y: pd.Series) -> TrainValTest:
    """Hold out 40%, then split that half into validation and test."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Split: train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}")
    return TrainValTest(X_train, y_train, X_val, y_val, X_test, y_test)


def score_model(model, parts: TrainValTest) -> dict:
    """Score one fitted model. The threshold comes from validation, then test is scored once."""
    val_proba = model.predict_proba(parts.X_val)[:, 1]
    test_proba = model.predict_proba(parts.X_test)[:, 1]
    threshold = threshold_for_recall(parts.y_val, val_proba, TARGET_RECALL)
    return {
        "validation": ranking_metrics(parts.y_val, val_proba),
        "test": ranking_metrics(parts.y_test, test_proba),
        "test_at_threshold": threshold_metrics(parts.y_test, test_proba, threshold),
        "_val_proba": val_proba,
        "_test_proba": test_proba,
    }


def fit_and_score(parts: TrainValTest, skip_tuning: bool) -> tuple[dict, dict, dict | None]:
    """Fit every candidate on train and record validation and test metrics."""
    fitted, results, best_params = {}, {}, None
    for name, model in candidate_models(SEED).items():
        print(f"Training {name}...")
        if name == "Gradient Boosting" and not skip_tuning:
            model, best_params = tune_gradient_boosting(model, parts.X_train, parts.y_train)
        else:
            model = clone(model).fit(parts.X_train, parts.y_train)
        fitted[name] = model
        results[name] = score_model(model, parts)
        r = results[name]
        print(
            f"  val PR-AUC {r['validation']['pr_auc']:.4f} | test ROC-AUC "
            f"{r['test']['roc_auc']:.4f} | test recall {r['test_at_threshold']['recall']:.3f} "
            f"@ precision {r['test_at_threshold']['precision']:.3f}"
        )
    return fitted, results, best_params


def choose_model(fitted: dict, results: dict) -> tuple[str, object, dict]:
    """Pick the model with the highest validation PR-AUC."""
    best_name = max(results, key=lambda name: results[name]["validation"]["pr_auc"])
    best = results[best_name]
    threshold = best["test_at_threshold"]["threshold"]
    print(f"Selected {best_name} (highest validation PR-AUC), threshold {threshold:.4f}")
    return best_name, fitted[best_name], best


def feature_importance(model, parts: TrainValTest) -> pd.DataFrame:
    """Shuffle each test column and measure the drop in PR-AUC."""
    print("Computing permutation importance...")
    sample = parts.X_test.sample(n=min(20_000, len(parts.X_test)), random_state=SEED)
    importance = permutation_importance(
        model,
        sample,
        parts.y_test.loc[sample.index],
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": importance.importances_mean,
                "std": importance.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def training_metadata(best_name, X, y, report, parts, best, best_params, importance_df, results) -> dict:
    """Build the JSON saved next to the model. Probabilities stay out of the file."""
    import sklearn

    train_df = parts.X_train.assign(target=parts.y_train.values)
    prevalence_by_group = (
        train_df.groupby(["Sex", "Age"])["target"].mean().round(4).reset_index().to_dict(orient="records")
    )
    return {
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%d"),
        "n_rows": int(len(X)),
        "cleaning": report.to_dict(),
        "split_sizes": {
            "train": len(parts.X_train),
            "validation": len(parts.X_val),
            "test": len(parts.X_test),
        },
        "prevalence": float(y.mean()),
        "target_recall": TARGET_RECALL,
        "threshold": best["test_at_threshold"]["threshold"],
        "best_params": best_params,
        "feature_columns": FEATURE_COLUMNS,
        "reference_profile": {column: float(parts.X_train[column].median()) for column in FEATURE_COLUMNS},
        "prevalence_by_group": prevalence_by_group,
        "feature_importance": importance_df.round(5).to_dict(orient="records"),
        "results": {
            name: {key: value for key, value in row.items() if not key.startswith("_")}
            for name, row in results.items()
        },
    }


def save_model_artifacts(best_model, metadata: dict, logistic_model) -> None:
    """Write the selected model, its metadata, and the logistic regression the browser scores."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "heart_risk_model.joblib", compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_to_builtin))
    web_path = write_web_model(logistic_model, metadata)
    print(f"Wrote {web_path} for the static demo.")


def save_figures(results, y, parts: TrainValTest, importance_df, best) -> None:
    """Draw the evaluation plots from the scored test and validation sets."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    threshold = best["test_at_threshold"]["threshold"]
    plot_curves(results, parts.y_test, y.mean())
    plot_calibration(results, parts.y_test)
    plot_importance(importance_df)
    plot_threshold(best["_val_proba"], parts.y_val, threshold)
    plot_confusion(best["test_at_threshold"]["confusion_matrix"], threshold)


def main(skip_tuning: bool) -> None:
    _configure_plots()
    start = time.time()

    X, y, report = load_data()
    print(report.format())
    write_cleaning_records(report)
    print(f"Clean rows {len(X):,}, prevalence {y.mean():.1%}")

    parts = split_train_val_test(X, y)
    fitted, results, best_params = fit_and_score(parts, skip_tuning)
    best_name, best_model, best = choose_model(fitted, results)
    importance_df = feature_importance(best_model, parts)
    metadata = training_metadata(best_name, X, y, report, parts, best, best_params, importance_df, results)
    save_model_artifacts(best_model, metadata, fitted["Logistic Regression"])
    save_figures(results, y, parts, importance_df, best)
    print(f"Done in {time.time() - start:.0f}s. Model, metadata and figures written.")


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj)}")


def plot_curves(results, y_test, prevalence):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["_test_proba"])
        precision, recall, _ = precision_recall_curve(y_test, r["_test_proba"])
        ax1.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC {r['test']['roc_auc']:.3f})")
        ax2.plot(recall, precision, color=COLORS[name], lw=2, label=f"{name} (AP {r['test']['pr_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (test set)")
    ax2.axhline(prevalence, color="k", ls="--", lw=1, label=f"Random guess ({prevalence:.1%})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test set)", ylim=(0, 1))
    ax1.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png")
    plt.close(fig)


def plot_calibration(results, y_test):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, r in results.items():
        frac_pos, mean_pred = calibration_curve(y_test, r["_test_proba"], n_bins=15, strategy="quantile")
        ax.plot(
            mean_pred,
            frac_pos,
            "o-",
            ms=3,
            color=COLORS[name],
            lw=1.5,
            label=f"{name} (Brier {r['test']['brier']:.4f})",
        )
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=1, label="Perfectly calibrated")
    ax.set(
        xlabel="Predicted risk",
        ylabel="Observed rate of heart disease",
        title="Calibration (test set)",
        xlim=(0, 0.6),
        ylim=(0, 0.6),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png")
    plt.close(fig)


def plot_importance(importance_df, top_n=12):
    top = importance_df.head(top_n).iloc[::-1]
    labels = [FEATURE_BY_NAME[f].label for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels, top["importance"], xerr=top["std"], color="#e8554e", alpha=0.85)
    ax.set(
        xlabel="Drop in PR-AUC when feature is shuffled",
        title="What drives the model (permutation importance)",
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


def plot_threshold(val_proba, y_val, chosen):
    thresholds = np.linspace(0.01, 0.6, 120)
    rows = [threshold_metrics(y_val, val_proba, t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thresholds, [r["recall"] for r in rows], lw=2, label="Recall (cases caught)")
    ax.plot(thresholds, [r["precision"] for r in rows], lw=2, label="Precision (flags that are real)")
    ax.plot(thresholds, [r["flagged_rate"] for r in rows], lw=2, ls=":", label="Share of people flagged")
    ax.axvline(chosen, color="k", ls="--", lw=1)
    ax.annotate(
        f"chosen threshold {chosen:.3f}\n(≥{TARGET_RECALL:.0%} recall on validation)",
        xy=(chosen, 0.9),
        xytext=(chosen + 0.04, 0.9),
        fontsize=8,
        va="center",
    )
    ax.set(
        xlabel="Decision threshold (predicted risk)",
        ylabel="Rate",
        title="Choosing a screening threshold (validation set)",
        ylim=(0, 1.02),
    )
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_tradeoff.png")
    plt.close(fig)


def plot_confusion(cm, threshold):
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        share = v / matrix[i].sum()
        ax.text(
            j,
            i,
            f"{v:,}\n({share:.0%} of row)",
            ha="center",
            va="center",
            color="white" if v > matrix.max() / 2 else "black",
            fontsize=9,
        )
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["No heart disease", "Heart disease"])
    ax.set(xlabel="Model decision", ylabel="Actual", title=f"Test set at threshold {threshold:.3f}")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-tuning", action="store_true", help="skip the gradient boosting search")
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="fit logistic regression and write web/model.json, leaving the selected model untouched",
    )
    args = parser.parse_args()
    if args.web_only:
        path = refresh_logistic_artifact()
        print(f"Wrote {path}. The gradient boosting artifact in models/ was left as-is.")
    else:
        main(args.skip_tuning)
