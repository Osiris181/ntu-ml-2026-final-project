"""Build competition CSVs from an actual phase/session template.

Model predictions use long rows: stock_id, date, predicted_y.
The output uses the template's id rows and date columns in their exact order.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd


def _iso_date(value: object) -> str:
    if isinstance(value, str):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"Date must be YYYY-MM-DD: {value!r}")
        return value
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None or timestamp != timestamp.normalize():
        raise ValueError(f"Prediction date must be a timezone-free calendar date: {value!r}")
    return timestamp.date().isoformat()


def load_submission_template(template_path: str | Path, target_ids: list[str]) -> tuple[list[str], list[str]]:
    """Return (stock IDs, date headers), validating the actual supplied template."""
    with Path(template_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.reader(file) if row]
    if not rows:
        raise ValueError("Empty submission template")
    header = rows[0]
    if len(header) < 2 or header[0] != "id":
        raise ValueError("Template must start with id and have at least one date column")
    dates = header[1:]
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate template date columns")
    for day in dates:
        _iso_date(day)
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError("Template rows have inconsistent column counts")
    ids = [row[0] for row in rows[1:]]
    if len(ids) != 50 or len(set(ids)) != 50 or any(not x or x.strip() != x for x in ids):
        raise ValueError("Template must contain 50 distinct, nonempty stock IDs")
    if len(target_ids) != 50 or len(set(target_ids)) != 50 or set(ids) != set(target_ids):
        raise ValueError("Template stock IDs must exactly match Top50@Y")
    return ids, dates


def build_submission(
    predictions: pd.DataFrame,
    template_path: str | Path,
    target_ids: list[str],
) -> pd.DataFrame:
    """Pivot complete long-form binary predictions into the template's layout."""
    ids, dates = load_submission_template(template_path, target_ids)
    required = {"stock_id", "date", "predicted_y"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions require columns: {sorted(required)}")
    selected = predictions.loc[:, ["stock_id", "date", "predicted_y"]].copy()
    if selected[["stock_id", "date", "predicted_y"]].isna().any().any():
        raise ValueError("Prediction keys and values cannot be missing")
    selected["stock_id"] = selected["stock_id"].astype(str)
    selected["date"] = selected["date"].map(_iso_date)
    if selected.duplicated(["stock_id", "date"]).any():
        raise ValueError("Duplicate prediction stock/date keys")
    if not selected["predicted_y"].isin([0, 1, "0", "1"]).all():
        raise ValueError("Predictions must be binary 0/1")
    selected["predicted_y"] = selected["predicted_y"].astype(int)
    values = {
        (row.stock_id, row.date): row.predicted_y
        for row in selected.itertuples(index=False)
    }
    expected = {(stock_id, day) for stock_id in ids for day in dates}
    if set(values) != expected:
        missing = len(expected - set(values))
        extra = len(set(values) - expected)
        raise ValueError(f"Predictions do not match template keys: {missing} missing, {extra} extra")
    return pd.DataFrame(
        [[stock_id, *(values[(stock_id, day)] for day in dates)] for stock_id in ids],
        columns=["id", *dates],
    )


def write_submission(
    predictions: pd.DataFrame,
    template_path: str | Path,
    output_path: str | Path,
    target_ids: list[str],
) -> Path:
    """Write a validated submission CSV without overwriting its template."""
    output = Path(output_path)
    if output.resolve() == Path(template_path).resolve():
        raise ValueError("Output path must not overwrite the submission template")
    if output.exists():
        raise FileExistsError(output)
    submission = build_submission(predictions, template_path, target_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, lineterminator="\n")
    return output
