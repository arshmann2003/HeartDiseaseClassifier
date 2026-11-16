import pandas as pd
import pytest

from heart_risk.cleaning import clean_survey
from heart_risk.features import FEATURE_COLUMNS, FEATURES, TARGET


def valid_row(**overrides) -> dict:
    row = {feature.name: feature.default for feature in FEATURES}
    row[TARGET] = 0
    row.update(overrides)
    return row


def test_valid_row_is_kept_in_model_column_order():
    raw = pd.DataFrame([valid_row(), valid_row(HighBP=1, **{TARGET: 1})])
    X, y, report = clean_survey(raw)

    assert list(X.columns) == FEATURE_COLUMNS
    assert X["HighBP"].tolist() == [0, 1]
    assert y.tolist() == [0, 1]
    assert report.rows_in == report.rows_out == 2
    assert report.missing.empty
    assert report.out_of_range.empty
    assert report.duplicates.empty
    assert list(raw.columns) == list(pd.DataFrame([valid_row()]).columns)


def test_missing_rows_are_stored_with_the_original_value():
    rows = [
        valid_row(),
        valid_row(**{TARGET: None}),
        valid_row(BMI=None),
        valid_row(BMI="not a number"),
        valid_row(HighBP="Yes"),
    ]
    X, y, report = clean_survey(pd.DataFrame(rows))

    assert len(X) == len(y) == 1
    assert report.dropped_missing == 4
    assert report.missing["HighBP"].tolist()[-1] == "Yes"
    assert "BMI" in report.missing["reason"].tolist()[2]
    assert report.rows_in == report.rows_out + report.dropped_missing + report.dropped_out_of_range


def test_out_of_range_rows_are_stored_on_their_own():
    rows = [
        valid_row(),
        valid_row(HighBP=2),
        valid_row(BMI=11),
        valid_row(BMI=999),
        valid_row(**{TARGET: 3}),
    ]
    X, _, report = clean_survey(pd.DataFrame(rows))

    assert len(X) == 1
    assert report.dropped_out_of_range == 4
    assert report.missing.empty
    assert set(report.out_of_range["HighBP"]) == {0, 2}
    assert "HighBP" in report.out_of_range.loc[report.out_of_range["HighBP"].eq(2), "reason"].item()


def test_a_bmi_inside_the_range_is_kept():
    X, _, report = clean_survey(pd.DataFrame([valid_row(BMI=27.4)]))

    assert report.rows_out == 1
    assert report.out_of_range.empty
    assert X["BMI"].item() == 27.4


def test_duplicate_rows_are_stored_and_kept():
    row = valid_row(Age=9, HighBP=1)
    _, _, report = clean_survey(pd.DataFrame([row, row, valid_row()]))

    assert report.rows_out == 3
    assert report.duplicate_rows_kept == 2
    assert len(report.duplicates) == 2
    assert set(report.duplicates["reason"]) == {"duplicate row kept"}
    assert "person id" in report.to_dict()["duplicate_decision"]


def test_extra_columns_are_ignored():
    row = valid_row()
    row["notes"] = "free text that is not a feature"
    X, _, report = clean_survey(pd.DataFrame([row]))

    assert report.rows_out == 1
    assert "notes" not in X.columns
    assert "notes" not in report.kept.columns


def test_missing_required_column_stops_the_run():
    frame = pd.DataFrame([valid_row()]).drop(columns=["BMI"])

    with pytest.raises(ValueError, match="BMI"):
        clean_survey(frame)
