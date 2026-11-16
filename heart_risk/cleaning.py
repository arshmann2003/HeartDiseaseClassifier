"""Check the survey file before any split or model fit.

Walk `clean_survey` from step 1 to step 4. Each step uses a pandas check,
keeps the raw row, and puts the result in its own table on the report.

Rows in = rows out + missing rows + out-of-range rows.
Duplicate rows are counted and kept, because this file has no person id.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FEATURE_BY_NAME, FEATURE_COLUMNS, TARGET

CHECKED = [TARGET, *FEATURE_COLUMNS]
DUPLICATE_DECISION = (
    "Duplicate rows are counted and kept. There is no person id, "
    "so two matching rows are two people."
)


@dataclass
class CleaningReport:
    """The audit returned with the cleaned table.

    `missing` and `out_of_range` hold the rows that were removed.
    `duplicates` holds the matching rows that were kept.
    """

    rows_in: int
    kept: pd.DataFrame
    missing: pd.DataFrame
    out_of_range: pd.DataFrame
    duplicates: pd.DataFrame

    @property
    def rows_out(self) -> int:
        return len(self.kept)

    @property
    def dropped_missing(self) -> int:
        return len(self.missing)

    @property
    def dropped_out_of_range(self) -> int:
        return len(self.out_of_range)

    @property
    def duplicate_rows_kept(self) -> int:
        return len(self.duplicates)

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_missing": self.dropped_missing,
            "dropped_out_of_range": self.dropped_out_of_range,
            "duplicate_rows_kept": self.duplicate_rows_kept,
            "duplicate_decision": DUPLICATE_DECISION,
            "rows_in_equals_out_plus_dropped": self.rows_in
            == self.rows_out + self.dropped_missing + self.dropped_out_of_range,
        }

    def format(self) -> str:
        return "\n".join(
            [
                "Data checks, before the split",
                f"  Step 1. Columns kept. Rows in: {self.rows_in:,}",
                f"  Step 2. Missing rows removed: {self.dropped_missing:,}",
                f"  Step 3. Out-of-range rows removed: {self.dropped_out_of_range:,}",
                f"  Step 4. Duplicate rows kept: {self.duplicate_rows_kept:,}",
                f"  {DUPLICATE_DECISION}",
                f"  Rows out: {self.rows_out:,}",
                "  Check: rows in = rows out + missing + out of range.",
            ]
        )


def clean_survey(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    """Run the four steps, then return the training table and the three record tables."""
    # Step 1. Keep the label and the 21 codebook columns.
    table = step_1_keep_required_columns(raw)
    # Step 2. Remove rows with a missing label or a missing answer.
    complete, missing = step_2_remove_missing_values(table)
    # Step 3. Remove rows with a value the codebook does not allow.
    valid, out_of_range = step_3_remove_out_of_range(complete)
    # Step 4. Record duplicate rows, and keep them.
    kept, duplicates = step_4_record_duplicate_rows(valid)
    report = CleaningReport(
        rows_in=len(table),
        kept=kept,
        missing=missing,
        out_of_range=out_of_range,
        duplicates=duplicates,
    )
    if not report.to_dict()["rows_in_equals_out_plus_dropped"]:
        raise RuntimeError("cleaned rows and removed rows do not add up to the rows that came in")
    features = kept[FEATURE_COLUMNS].reset_index(drop=True)
    labels = kept[TARGET].reset_index(drop=True)
    return features, labels, report


def step_1_keep_required_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy the required columns. The caller's frame is left unchanged."""
    required = [TARGET, *FEATURE_COLUMNS]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError("survey file is missing required columns: " + ", ".join(missing))
    table = raw.loc[:, required].copy()
    table.insert(0, "source_row", raw.index)
    return table


def step_2_remove_missing_values(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn non-numbers into missing, then split those rows into their own table."""
    numbers = table.copy()
    numbers[CHECKED] = table[CHECKED].apply(pd.to_numeric, errors="coerce")
    missing_cells = numbers[CHECKED].isna()
    missing_rows = missing_cells.any(axis=1)
    removed = table.loc[missing_rows].copy()
    removed["reason"] = _row_reasons(missing_cells.loc[missing_rows])
    return numbers.loc[~missing_rows].copy(), removed


def step_3_remove_out_of_range(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off values that are not in the codebook. Yes/no must be 0 or 1."""
    bad = pd.DataFrame(False, index=table.index, columns=CHECKED)
    bad[TARGET] = ~table[TARGET].isin([0, 1])
    for name in FEATURE_COLUMNS:
        feature = FEATURE_BY_NAME[name]
        column = table[name]
        if feature.options:
            bad[name] = ~column.isin(list(feature.options))
        else:
            bad[name] = ~column.between(feature.min_value, feature.max_value)
    bad_rows = bad.any(axis=1)
    removed = table.loc[bad_rows].copy()
    removed["reason"] = _row_reasons(bad.loc[bad_rows])
    return table.loc[~bad_rows].copy(), removed


def step_4_record_duplicate_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find rows with the same answers. Keep them, and store a copy."""
    matches = table.duplicated(subset=CHECKED, keep=False)
    duplicates = table.loc[matches].copy()
    duplicates["reason"] = "duplicate row kept"
    return table.copy(), duplicates


def _row_reasons(flags: pd.DataFrame) -> pd.Series:
    """Name the columns that failed, one string per row."""
    if flags.empty:
        return pd.Series(dtype="object")
    return flags.apply(lambda row: ", ".join(flags.columns[row]), axis=1)
