"""Streamlit web demo for the heart disease risk screening model.

streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from heart_risk.features import FEATURE_BY_NAME, FEATURE_COLUMNS, FEATURES, YES_NO, describe_value
from heart_risk.modeling import explain_prediction

ROOT = Path(__file__).parent
REPO_URL = "https://github.com/arshmann2003/HeartDiseaseClassifier"
GROUPS = ["Demographics", "Medical history", "Lifestyle", "Health status", "Healthcare access"]
SLIDER_ORDINALS = {"Age", "GenHlth"}
RADIO_ORDINALS = {"Diabetes"}

PRESETS = {
    "Healthy 35-year-old": {
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
    "55-year-old with high BP": {
        **{f.name: f.default for f in FEATURES},
        "Sex": 1,
        "HighBP": 1,
        "HighChol": 1,
        "BMI": 29,
        "PhysActivity": 0,
    },
    "High-risk 70-year-old": {
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
}

st.set_page_config(page_title="Heart Disease Risk Screener", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "heart_risk_model.joblib")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, metadata


def is_toggle(feature) -> bool:
    return feature.kind == "binary" and feature.options == YES_NO


def apply_profile(profile: dict[str, int]) -> None:
    for name, value in profile.items():
        st.session_state[name] = bool(value) if is_toggle(FEATURE_BY_NAME[name]) else value


def render_input(feature) -> None:
    kwargs = {"key": feature.name, "help": feature.help or None}
    if is_toggle(feature):
        st.toggle(feature.label, **kwargs)
    elif feature.kind == "numeric" and feature.name == "BMI":
        st.number_input(feature.label, feature.min_value, feature.max_value, step=1, **kwargs)
    elif feature.kind == "numeric":
        st.slider(feature.label, feature.min_value, feature.max_value, **kwargs)
    elif feature.name in SLIDER_ORDINALS:
        st.select_slider(feature.label, list(feature.options), format_func=feature.options.get, **kwargs)
    elif feature.kind == "binary" or feature.name in RADIO_ORDINALS:
        st.radio(
            feature.label, list(feature.options), format_func=feature.options.get, horizontal=True, **kwargs
        )
    else:
        st.selectbox(feature.label, list(feature.options), format_func=feature.options.get, **kwargs)


def current_profile() -> pd.DataFrame:
    return pd.DataFrame([{name: int(st.session_state[name]) for name in FEATURE_COLUMNS}])


def group_prevalence(metadata, sex: int, age: int) -> float:
    for row in metadata["prevalence_by_group"]:
        if row["Sex"] == sex and row["Age"] == age:
            return row["target"]
    return metadata["prevalence"]


def drivers_chart(contributions: pd.DataFrame, profile: pd.DataFrame, top_n: int = 8):
    top = contributions[contributions["contribution"].abs() >= 0.001].head(top_n).copy()
    if top.empty:
        return None
    top["factor"] = [
        f"{FEATURE_BY_NAME[f].label}: {describe_value(f, profile.at[0, f])}" for f in top["feature"]
    ]
    top["points"] = top["contribution"] * 100
    top["effect"] = top["points"].map(lambda p: "Raises risk" if p > 0 else "Lowers risk")
    return (
        alt.Chart(top)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("points:Q", title="Risk change (percentage points)"),
            y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#e8554e", "#3a9d6b"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("factor:N", title="Factor"),
                alt.Tooltip("points:Q", title="Percentage points", format="+.1f"),
            ],
        )
        .properties(height=36 * len(top) + 40)
    )


def comparison_chart(risk: float, peer: float, overall: float, threshold: float):
    data = pd.DataFrame(
        {
            "who": ["This profile", "Same age group and sex", "All US adults surveyed"],
            "risk": [risk, peer, overall],
        }
    )
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X(
                "risk:Q",
                title="Share with heart disease",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(0.25, risk, peer) * 1.15]),
            ),
            y=alt.Y("who:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.condition(alt.datum.who == "This profile", alt.value("#e8554e"), alt.value("#b8c2cc")),
            tooltip=[alt.Tooltip("who:N", title=""), alt.Tooltip("risk:Q", title="Risk", format=".1%")],
        )
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("risk:Q", format=".1%"), color=alt.value("#333")
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 3], color="#555").encode(x="t:Q")
    )
    return (bars + labels + rule).properties(height=150)


def estimator_tab(model, metadata) -> None:
    threshold = metadata["threshold"]
    at = metadata["results"][metadata["model_name"]]["test_at_threshold"]
    lift = at["precision"] / metadata["prevalence"]
    st.write("Try an example profile, or fill in your own answers below.")
    cols = st.columns(len(PRESETS))
    for col, (name, profile) in zip(cols, PRESETS.items(), strict=True):
        col.button(name, on_click=apply_profile, args=(profile,), width="stretch")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        for group in GROUPS:
            with st.container(border=True):
                st.markdown(f"**{group}**")
                for feature in (f for f in FEATURES if f.group == group):
                    render_input(feature)

    profile = current_profile()
    risk = float(model.predict_proba(profile)[0, 1])
    peer = group_prevalence(metadata, profile.at[0, "Sex"], profile.at[0, "Age"])

    with right:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("Estimated risk", f"{risk:.1%}")
            c2.metric(
                "Average for age and sex",
                f"{peer:.1%}",
                delta=f"{(risk - peer) * 100:+.1f} pts",
                delta_color="inverse",
            )
            if risk >= threshold:
                st.error(
                    f"**Flagged for follow-up.** The estimate is above the {threshold:.1%} screening "
                    f"threshold. In testing, people above this line were {lift:.1f}x more likely than "
                    "average to have heart disease."
                )
            else:
                st.success(
                    f"**Below the screening threshold** of {threshold:.1%}. In testing, "
                    f"{at['recall']:.0%} of people with heart disease scored above this line."
                )
            st.altair_chart(
                comparison_chart(risk, peer, metadata["prevalence"], threshold),
                width="stretch",
            )
            st.caption("Dashed line marks the screening threshold.")

        with st.container(border=True):
            st.markdown("**What's driving this estimate**")
            st.caption(
                "Each bar shows how much the estimate changes if that one answer is replaced "
                "with the most common answer in the survey. These are statistical associations, "
                "not causes."
            )
            chart = drivers_chart(explain_prediction(model, profile, metadata["reference_profile"]), profile)
            if chart is None:
                st.write("This profile is close to the typical respondent on every factor.")
            else:
                st.altair_chart(chart, width="stretch")


def performance_tab(metadata) -> None:
    best = metadata["results"][metadata["model_name"]]
    at = best["test_at_threshold"]
    cm = at["confusion_matrix"]
    st.markdown(
        f"All numbers are from a **held-out test set of {metadata['split_sizes']['test']:,} people** "
        "that was not used for training, model selection or choosing the threshold."
    )
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{best['test']['roc_auc']:.3f}")
    c[1].metric(
        "PR-AUC", f"{best['test']['pr_auc']:.3f}", help=f"Random guessing scores {metadata['prevalence']:.3f}"
    )
    c[2].metric("Recall at threshold", f"{at['recall']:.0%}", help="Share of true cases flagged")
    c[3].metric("Precision at threshold", f"{at['precision']:.0%}", help="Share of flags that are true cases")

    one_in = round(1 / at["precision"])
    lift = at["precision"] / metadata["prevalence"]
    st.markdown(
        f"In plain terms: the model flags **{at['flagged_rate']:.0%}** of people and catches "
        f"**{cm['tp']:,} of {cm['tp'] + cm['fn']:,}** people with heart disease. About 1 in {one_in} "
        f"flagged people actually has heart disease, which is {lift:.1f}x the base rate."
        " For a cheap first-pass screen that is followed by a proper check-up, that trade-off "
        "is reasonable; for a diagnosis it would not be."
    )

    rows = []
    for name, r in metadata["results"].items():
        rows.append(
            {
                "Model": name + (" (deployed)" if name == metadata["model_name"] else ""),
                "ROC-AUC": r["test"]["roc_auc"],
                "PR-AUC": r["test"]["pr_auc"],
                "Brier score": r["test"]["brier"],
                "Recall": r["test_at_threshold"]["recall"],
                "Precision": r["test_at_threshold"]["precision"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Brier score": "{:.4f}",
                "Recall": "{:.1%}",
                "Precision": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    figures = ROOT / "reports" / "figures"
    st.image(str(figures / "roc_pr_curves.png"))
    a, b = st.columns(2)
    a.image(str(figures / "threshold_tradeoff.png"))
    b.image(str(figures / "calibration.png"))
    a.image(str(figures / "feature_importance.png"))
    b.image(str(figures / "confusion_matrix.png"))


def about_tab(metadata) -> None:
    st.markdown(f"""
**Data.** {metadata["n_rows"]:,} responses to the CDC's 2015 Behavioral Risk Factor Surveillance
System (BRFSS) telephone survey of US adults. The target is whether the respondent reported
ever having coronary heart disease or a heart attack ({metadata["prevalence"]:.1%} said yes).

**Method.** The data is split 60/20/20 into train, validation and test sets. Logistic regression,
random forest and gradient boosting are compared. Gradient boosting hyperparameters are tuned with
cross-validated random search on the training set. The model with the best validation PR-AUC is
deployed. The decision threshold is the highest one that reaches {metadata["target_recall"]:.0%}
recall on validation data. No class weighting is used, so predicted risks stay calibrated.

**Limitations.**
- Answers are self-reported, and the survey covers US adults in 2015 only.
- The target is *having had* heart disease, not developing it in the future. Some inputs, like
  difficulty walking, may be consequences of heart disease rather than causes.

Source code, training pipeline and tests: [{REPO_URL.removeprefix("https://")}]({REPO_URL})
""")


def main() -> None:
    model, metadata = load_artifacts()
    for feature in FEATURES:
        if feature.name not in st.session_state:
            apply_profile({feature.name: feature.default})

    st.title("Heart Disease Risk Screener")
    st.markdown(
        f"A machine learning model trained on **{metadata['n_rows']:,} CDC health survey responses** "
        "estimates the chance that someone with a given profile has heart disease. "
        f"[View the code on GitHub]({REPO_URL})"
    )
    st.caption("For education and demonstration only. Not medical advice.")

    estimator, performance, about = st.tabs(["Risk estimator", "Model performance", "About"])
    with estimator:
        estimator_tab(model, metadata)
    with performance:
        performance_tab(metadata)
    with about:
        about_tab(metadata)


main()
