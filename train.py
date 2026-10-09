"""Train, evaluate and export the heart disease risk screening model.

    python train.py                  # full pipeline (includes hyperparameter search)
    python train.py --skip-tuning    # faster, uses default gradient boosting settings

Data is split 60/20/20 into train / validation / test. Model selection,
hyperparameter search and the decision threshold only ever see train and
validation data; the test set is scored once at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
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

SEED = 42
TARGET_RECALL = 0.80
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")
COLORS = {"Logistic Regression": "#8c8c8c", "Random Forest": "#4c9be8", "Gradient Boosting": "#e8554e"}

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


def main(skip_tuning: bool) -> None:
    start = time.time()
    X, y = load_data()
    print(f"Loaded {len(X):,} rows, prevalence {y.mean():.1%}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Split: train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}")

    fitted, results, best_params = {}, {}, None
    for name, model in candidate_models(SEED).items():
        print(f"Training {name}...")
        if name == "Gradient Boosting" and not skip_tuning:
            model, best_params = tune_gradient_boosting(model, X_train, y_train)
        else:
            model = clone(model).fit(X_train, y_train)
        fitted[name] = model

        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        threshold = threshold_for_recall(y_val, val_proba, TARGET_RECALL)
        results[name] = {
            "validation": ranking_metrics(y_val, val_proba),
            "test": ranking_metrics(y_test, test_proba),
            "test_at_threshold": threshold_metrics(y_test, test_proba, threshold),
            "_val_proba": val_proba,
            "_test_proba": test_proba,
        }
        r = results[name]
        print(
            f"  val PR-AUC {r['validation']['pr_auc']:.4f} | test ROC-AUC "
            f"{r['test']['roc_auc']:.4f} | test recall {r['test_at_threshold']['recall']:.3f} "
            f"@ precision {r['test_at_threshold']['precision']:.3f}"
        )

    best_name = max(results, key=lambda n: results[n]["validation"]["pr_auc"])
    best_model = fitted[best_name]
    best = results[best_name]
    threshold = best["test_at_threshold"]["threshold"]
    print(f"Selected {best_name} (highest validation PR-AUC), threshold {threshold:.4f}")

    print("Computing permutation importance...")
    sample = X_test.sample(n=min(20_000, len(X_test)), random_state=SEED)
    importance = permutation_importance(
        best_model,
        sample,
        y_test.loc[sample.index],
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_df = (
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

    train_df = X_train.assign(target=y_train.values)
    prevalence_by_group = (
        train_df.groupby(["Sex", "Age"])["target"].mean().round(4).reset_index().to_dict(orient="records")
    )
    reference_profile = {c: float(X_train[c].median()) for c in FEATURE_COLUMNS}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "heart_risk_model.joblib", compress=3)

    import sklearn

    metadata = {
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%d"),
        "n_rows": int(len(X)),
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "prevalence": float(y.mean()),
        "target_recall": TARGET_RECALL,
        "threshold": threshold,
        "best_params": best_params,
        "feature_columns": FEATURE_COLUMNS,
        "reference_profile": reference_profile,
        "prevalence_by_group": prevalence_by_group,
        "feature_importance": importance_df.round(5).to_dict(orient="records"),
        "results": {
            name: {k: v for k, v in r.items() if not k.startswith("_")} for name, r in results.items()
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_to_builtin))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(results, y_test, y.mean())
    plot_calibration(results, y_test)
    plot_importance(importance_df)
    plot_threshold(best["_val_proba"], y_val, threshold)
    plot_confusion(best["test_at_threshold"]["confusion_matrix"], threshold)
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
    main(parser.parse_args().skip_tuning)
