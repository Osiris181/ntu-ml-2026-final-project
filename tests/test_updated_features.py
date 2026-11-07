"""Checks for using later source files without turning them into labels."""

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared.data_loader import GLOBAL_COLUMNS, MARKET_COLUMNS
from shared.features import build_f0_dataset
from shared.updated_features import add_template_keys, combine_feature_history, load_updated_market


def market_rows(ids, dates):
    rows = []
    for stock_id in ids:
        for position, day in enumerate(dates):
            row = {column: 1.0 for column in MARKET_COLUMNS if column != "y"}
            row.update({
                "stock_id": stock_id,
                "date": pd.Timestamp(day),
                "etl:adj_open": 10.0 + position,
                "etl:adj_close": 11.0 + position,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def global_rows(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        **{column: [float(i + 1) for i in range(len(dates))] for column in GLOBAL_COLUMNS},
    })


def write_template(path, ids, dates):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", *dates])
        writer.writerows([[stock_id, *([0] * len(dates))] for stock_id in ids])


class UpdatedFeaturesTests(unittest.TestCase):
    def test_updated_labels_are_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "market").mkdir()
            frame = pd.DataFrame([
                {"date": "2026-09-01", **{column: 1.0 for column in MARKET_COLUMNS}},
                {"date": "2026-09-02", **{column: 2.0 for column in MARKET_COLUMNS}},
            ])
            frame.loc[1, "y"] = -1
            frame.to_csv(root / "market" / "2330.csv", index=False)
            loaded = load_updated_market(root, ["2330"])
            self.assertEqual(len(loaded), 2)
            self.assertNotIn("y", loaded)
            self.assertEqual(loaded.loc[1, "etl:adj_close"], 2.0)

    def test_append_keeps_original_history_and_rejects_overlap(self):
        original = market_rows(["2330"], ["2026-08-31"])
        updated = market_rows(["2330"], ["2026-09-01"])
        original_global = global_rows(["2026-08-31"])
        updated_global = global_rows(["2026-09-01"])
        combined, globals_combined = combine_feature_history(
            original, updated, original_global, updated_global
        )
        self.assertEqual(combined["date"].tolist(), pd.to_datetime(["2026-08-31", "2026-09-01"]).tolist())
        self.assertEqual(len(globals_combined), 2)
        with self.assertRaisesRegex(ValueError, "start after"):
            combine_feature_history(original, original, original_global, updated_global)

    def test_template_rows_use_only_prior_values(self):
        ids = [str(i) for i in range(50)]
        market = market_rows(ids, ["2026-08-31", "2026-09-01"])
        with tempfile.TemporaryDirectory() as temp:
            template = Path(temp) / "template.csv"
            write_template(template, ids, ["2026-09-02"])
            extended = add_template_keys(market, template, ids)
            self.assertEqual(len(extended), 150)
            future = extended.loc[extended.date.eq("2026-09-02")]
            self.assertEqual(len(future), 50)
            self.assertTrue(future["etl:adj_close"].isna().all())
            f0 = build_f0_dataset(extended, global_rows(["2026-08-31", "2026-09-01"]))
            row = f0.loc[f0.stock_id.eq("0") & f0.date.eq("2026-09-02")].iloc[0]
            self.assertAlmostEqual(row.intraday_return_1d, 12 / 11 - 1)
            self.assertEqual(row.day_of_week, "Wednesday")
            self.assertNotIn("y", f0)

    def test_missing_inside_history_and_multiple_future_dates_fail(self):
        ids = [str(i) for i in range(50)]
        market = market_rows(ids, ["2026-09-01", "2026-09-03"])
        with tempfile.TemporaryDirectory() as temp:
            template = Path(temp) / "template.csv"
            write_template(template, ids, ["2026-09-02"])
            with self.assertRaisesRegex(ValueError, "absent inside observed"):
                add_template_keys(market, template, ids)
            write_template(template, ids, ["2026-09-04", "2026-09-07"])
            with self.assertRaisesRegex(ValueError, "Multiple future dates"):
                add_template_keys(market, template, ids)


if __name__ == "__main__":
    unittest.main()
