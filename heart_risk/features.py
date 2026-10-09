"""Feature schema for the BRFSS 2015 heart disease health indicators dataset.

Every encoding below follows the CDC BRFSS 2015 codebook as used by the Kaggle
release of the dataset, so the training pipeline and the web app agree on what
each integer means.
"""

from dataclasses import dataclass, field

TARGET = "HeartDiseaseorAttack"


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str  # "binary", "ordinal" or "numeric"
    group: str
    help: str = ""
    options: dict[int, str] = field(default_factory=dict)
    min_value: int | None = None
    max_value: int | None = None
    default: int = 0


YES_NO = {0: "No", 1: "Yes"}

AGE_BUCKETS = {
    1: "18–24",
    2: "25–29",
    3: "30–34",
    4: "35–39",
    5: "40–44",
    6: "45–49",
    7: "50–54",
    8: "55–59",
    9: "60–64",
    10: "65–69",
    11: "70–74",
    12: "75–79",
    13: "80+",
}

EDUCATION_LEVELS = {
    1: "Never attended school / kindergarten only",
    2: "Grades 1–8",
    3: "Grades 9–11",
    4: "High school graduate / GED",
    5: "Some college or technical school",
    6: "College graduate",
}

INCOME_BRACKETS = {
    1: "Less than $10k",
    2: "$10k–$15k",
    3: "$15k–$20k",
    4: "$20k–$25k",
    5: "$25k–$35k",
    6: "$35k–$50k",
    7: "$50k–$75k",
    8: "$75k or more",
}

GENERAL_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

DIABETES_STATUS = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}

SEX = {0: "Female", 1: "Male"}


FEATURES: list[Feature] = [
    Feature("Age", "Age group", "ordinal", "Demographics", options=AGE_BUCKETS, default=8),
    Feature("Sex", "Sex", "binary", "Demographics", options=SEX, default=0),
    Feature("Education", "Education", "ordinal", "Demographics", options=EDUCATION_LEVELS, default=5),
    Feature("Income", "Household income", "ordinal", "Demographics", options=INCOME_BRACKETS, default=6),
    Feature(
        "HighBP",
        "High blood pressure",
        "binary",
        "Medical history",
        "Ever told by a health professional that you have high blood pressure.",
        YES_NO,
    ),
    Feature(
        "HighChol",
        "High cholesterol",
        "binary",
        "Medical history",
        "Ever told by a health professional that your cholesterol is high.",
        YES_NO,
    ),
    Feature(
        "CholCheck",
        "Cholesterol check in last 5 years",
        "binary",
        "Medical history",
        options=YES_NO,
        default=1,
    ),
    Feature("Diabetes", "Diabetes", "ordinal", "Medical history", options=DIABETES_STATUS),
    Feature("Stroke", "Ever had a stroke", "binary", "Medical history", options=YES_NO),
    Feature(
        "BMI",
        "Body mass index (BMI)",
        "numeric",
        "Lifestyle",
        "Weight (kg) divided by height (m) squared.",
        min_value=12,
        max_value=98,
        default=27,
    ),
    Feature(
        "Smoker",
        "Smoked 100+ cigarettes in lifetime",
        "binary",
        "Lifestyle",
        "About 5 packs over your whole life.",
        YES_NO,
    ),
    Feature(
        "PhysActivity",
        "Physically active in last 30 days",
        "binary",
        "Lifestyle",
        "Any exercise outside of your job.",
        YES_NO,
        default=1,
    ),
    Feature("Fruits", "Eat fruit at least once a day", "binary", "Lifestyle", options=YES_NO, default=1),
    Feature(
        "Veggies", "Eat vegetables at least once a day", "binary", "Lifestyle", options=YES_NO, default=1
    ),
    Feature(
        "HvyAlcoholConsump",
        "Heavy drinker",
        "binary",
        "Lifestyle",
        "Men: more than 14 drinks per week. Women: more than 7.",
        YES_NO,
    ),
    Feature(
        "GenHlth",
        "General health",
        "ordinal",
        "Health status",
        "How would you describe your health in general?",
        GENERAL_HEALTH,
        default=3,
    ),
    Feature(
        "PhysHlth",
        "Days of poor physical health (last 30)",
        "numeric",
        "Health status",
        min_value=0,
        max_value=30,
    ),
    Feature(
        "MentHlth",
        "Days of poor mental health (last 30)",
        "numeric",
        "Health status",
        min_value=0,
        max_value=30,
    ),
    Feature(
        "DiffWalk", "Serious difficulty walking or climbing stairs", "binary", "Health status", options=YES_NO
    ),
    Feature(
        "AnyHealthcare", "Have any health coverage", "binary", "Healthcare access", options=YES_NO, default=1
    ),
    Feature(
        "NoDocbcCost",
        "Skipped a doctor visit due to cost (last 12 months)",
        "binary",
        "Healthcare access",
        options=YES_NO,
    ),
]

FEATURE_BY_NAME = {f.name: f for f in FEATURES}

# Column order the model is trained on; matches the raw CSV.
FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "Diabetes",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

assert set(FEATURE_COLUMNS) == set(FEATURE_BY_NAME), "schema and column order disagree"


def describe_value(name: str, value: int) -> str:
    """Human-readable form of an encoded feature value."""
    feature = FEATURE_BY_NAME[name]
    if feature.options:
        return feature.options.get(int(value), str(value))
    return str(int(value))
