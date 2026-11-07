import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data_loader import load_metadata, validate_keys
from shared.evaluation import cv_score, e1_seed_summary, evaluate_predictions
from shared.features import (
    DATASET_COLUMNS, build_f0_dataset, load_f0_dataset,
    stock_history_candidates,
)
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import E1_SEEDS, FOLDS, HOLDOUT, split_market
from shared.submission import build_submission, load_submission_template, write_submission


def stock_rows(stock_id="2330", count=30):
    dates = pd.bdate_range("2020-01-01", periods=count)
    n = np.arange(1, count + 1, dtype=float)
    return pd.DataFrame({
        "stock_id": stock_id, "date": dates,
        "etl:adj_close": n + 10, "etl:adj_open": n + 9,
        "etl:adj_high": n + 11, "etl:adj_low": n + 8,
        "price:成交筆數": n + 20, "price:成交金額": n + 100,
        "price:成交股數": n + 200,
        "foreign_investors_shareholding:全體外資及陸資持股比率": n,
        "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)": n,
        "institutional_investors_trading_summary:投信買賣超股數": n,
        "RSI": n, "SMA": n + 5, "MACD": n,
        "ATR": n, "BBANDS_upper": n + 4,
        "BBANDS_mid": n + 3, "BBANDS_lower": n + 2,
    })


class SharedTests(unittest.TestCase):
    def test_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_keys(pd.DataFrame({"stock_id": ["A", "A"], "date": [pd.Timestamp("2020-01-01")] * 2}))

    def test_split_windows_and_target_only(self):
        dates = pd.to_datetime(["2022-12-30", "2023-01-02", "2024-01-02", "2025-01-02", "2026-01-02"])
        frame = pd.DataFrame([(stock, day) for day in dates for stock in [*(str(i) for i in range(50)), "extra"]], columns=["stock_id", "date"])
        target = [str(i) for i in range(50)]
        for split in (*FOLDS, HOLDOUT):
            train, evaluate = split_market(frame, target, split)
            self.assertLess(train.date.max(), evaluate.date.min())
            self.assertEqual(set(evaluate.stock_id), set(target))
            self.assertEqual(len(evaluate), 50)
            self.assertIn("extra", set(train.stock_id))

    def test_lags_and_rolling_are_causal(self):
        market = stock_rows()
        original = stock_history_candidates(market)
        self.assertEqual(original.loc[20, "source_date"], market.loc[19, "date"])
        self.assertEqual(original.loc[20, "institutional_source_date"], market.loc[18, "date"])
        expected = market.loc[19, "price:成交股數"] / market.loc[:19, "price:成交股數"].mean()
        self.assertAlmostEqual(original.loc[20, "relative_volume_20d"], expected)
        self.assertAlmostEqual(
            original.loc[20, "foreign_ownership"],
            market.loc[18, "foreign_investors_shareholding:全體外資及陸資持股比率"] / 100,
        )
        self.assertAlmostEqual(
            original.loc[20, "foreign_net_buy_ratio"],
            market.loc[18, "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)"]
            / market.loc[18, "price:成交股數"],
        )
        self.assertAlmostEqual(
            original.loc[20, "trust_net_buy_ratio"],
            market.loc[18, "institutional_investors_trading_summary:投信買賣超股數"]
            / market.loc[18, "price:成交股數"],
        )
        self.assertAlmostEqual(original.loc[20, "rsi_normalized"], market.loc[19, "RSI"] / 100)
        changed = market.copy()
        changed.loc[20:, "etl:adj_close"] = 100000
        changed.loc[20:, "price:成交股數"] = 100000
        changed.loc[20:, "RSI"] = 100000
        changed.loc[20:, "institutional_investors_trading_summary:投信買賣超股數"] = 100000
        revised = stock_history_candidates(changed)
        for field in ("intraday_return_1d", "relative_volume_20d", "rsi_normalized", "trust_net_buy_ratio"):
            self.assertAlmostEqual(original.loc[20, field], revised.loc[20, field])
        self.assertTrue(original.loc[19, "relative_volume_20d"] != original.loc[19, "relative_volume_20d"])
        self.assertTrue(pd.isna(original.loc[1, "foreign_net_buy_ratio"]))
        self.assertTrue(set(F0_FEATURES).isdisjoint({"y", "etl:adj_close", "etl:adj_open"}))

    def test_global_alignment_is_independent_and_bounded(self):
        market = stock_rows(count=10)
        global_data = pd.DataFrame({
            "date": pd.to_datetime(["2019-12-31", "2020-01-03", "2020-01-06"]),
            "^VIX": [10.0, 20.0, np.nan],
            "^IXIC": [100.0, 110.0, 121.0],
            "^GSPC": [200.0, np.nan, 220.0],
            "^TWII": [300.0, 330.0, np.nan],
        })
        f0 = build_f0_dataset(market, global_data)
        jan3 = f0.loc[f0.date.eq("2020-01-03")].iloc[0]
        self.assertAlmostEqual(jan3.vix_normalized, 0.1)  # same-date 20 is excluded
        self.assertTrue(pd.isna(jan3.nasdaq_return))
        jan6 = f0.loc[f0.date.eq("2020-01-06")].iloc[0]
        self.assertAlmostEqual(jan6.vix_normalized, 0.2)
        self.assertAlmostEqual(jan6.nasdaq_return, 0.1)
        self.assertTrue(pd.isna(jan6.sp500_return))  # this series has no prior valid return yet
        jan8 = f0.loc[f0.date.eq("2020-01-08")].iloc[0]
        self.assertAlmostEqual(jan8.vix_normalized, 0.2)  # exactly five days old
        self.assertAlmostEqual(jan8.sp500_return, 0.1)
        self.assertAlmostEqual(jan8.taiex_return, 0.1)
        jan9 = f0.loc[f0.date.eq("2020-01-09")].iloc[0]
        self.assertTrue(pd.isna(jan9.vix_normalized))  # Jan 3 is six calendar days old
        self.assertAlmostEqual(jan9.nasdaq_return, 0.1)  # Jan 6 remains eligible

    def test_f0_rows_are_complete_consistent_and_target_free(self):
        stock_b = stock_rows("B", 25).drop(index=5)  # B misses one trading date
        market = pd.concat([stock_rows("A", 30), stock_b], ignore_index=True)
        global_data = pd.DataFrame({
            "date": pd.to_datetime(["2019-12-31", "2020-01-02"]),
            "^VIX": [10.0, 11.0], "^IXIC": [100.0, 110.0],
            "^GSPC": [200.0, 210.0], "^TWII": [300.0, 310.0],
        })
        f0 = build_f0_dataset(market, global_data)
        self.assertEqual(len(f0), len(market))
        self.assertEqual(f0.columns.tolist(), list(DATASET_COLUMNS))
        self.assertEqual(len(F0_FEATURES), 21)
        self.assertEqual(set(map(tuple, f0[["stock_id", "date"]].to_numpy())),
                         set(map(tuple, market[["stock_id", "date"]].to_numpy())))
        self.assertFalse({"y", "etl:adj_close", "etl:adj_open", "source_date"} & set(f0.columns))
        self.assertEqual(f0.loc[0, "day_of_week"], f0.loc[0, "date"].day_name())
        self.assertTrue(pd.isna(f0.loc[0, "intraday_return_1d"]))
        b_jan9 = f0.loc[f0.stock_id.eq("B") & f0.date.eq("2020-01-09")].iloc[0]
        self.assertAlmostEqual(
            b_jan9.intraday_return_1d,
            stock_b.loc[4, "etl:adj_close"] / stock_b.loc[4, "etl:adj_open"] - 1,
        )
        pd.testing.assert_frame_equal(f0, build_f0_dataset(market.sample(frac=1, random_state=42), global_data))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "f0.csv"
            f0.to_csv(output, index=False)
            pd.testing.assert_frame_equal(f0, load_f0_dataset(output), check_freq=False)

    def test_preprocessing_fits_training_only(self):
        train = pd.DataFrame({field: [1.0, 3.0, np.inf] for field in NUMERIC_FEATURES})
        train["stock_id"] = ["A", "A", "B"]
        train["day_of_week"] = ["Monday", "Tuesday", None]
        processor = BaselinePreprocessor().fit(train)
        test = train.iloc[[0]].copy()
        test.loc[test.index[0], NUMERIC_FEATURES[0]] = 1e9
        test.loc[test.index[0], "stock_id"] = "C"
        transformed = processor.transform(test)
        self.assertEqual(processor.medians_[NUMERIC_FEATURES[0]], 2.0)
        self.assertEqual(processor.means_[NUMERIC_FEATURES[0]], 2.0)
        self.assertEqual(transformed[["stock_id=A", "stock_id=B"]].sum(axis=1).iloc[0], 0)
        self.assertIn("day_of_week=Unknown", transformed)
        restored = BaselinePreprocessor.from_state(json.loads(json.dumps(processor.to_state())))
        pd.testing.assert_frame_equal(transformed, restored.transform(test))
        changed_state = processor.to_state()
        changed_state["feature_names"] = []
        with self.assertRaisesRegex(ValueError, "output columns changed"):
            BaselinePreprocessor.from_state(changed_state)
        empty = train.copy()
        empty[NUMERIC_FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "All-missing"):
            BaselinePreprocessor().fit(empty)

    def test_official_score_and_complete_keys(self):
        evaluate = pd.DataFrame({
            "stock_id": ["A", "B"], "date": pd.to_datetime(["2023-01-02"] * 2),
            "etl:adj_open": [10.0, 20.0], "etl:adj_close": [11.0, 18.0], "y": [1, 0],
        })
        prediction = evaluate[["stock_id", "date"]].copy()
        prediction["predicted_y"] = [1, 0]
        measured = evaluate_predictions(evaluate, prediction)
        self.assertAlmostEqual(measured["official_score"], 0.1)
        self.assertEqual(measured["n_eval"], 2)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            evaluate_predictions(evaluate, prediction.iloc[:1])
        bad = evaluate.copy()
        bad.loc[0, "etl:adj_open"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_predictions(bad, prediction)
        self.assertAlmostEqual(cv_score({"2023": 1, "2024": 2, "2025": 3}), 2)

    def test_five_seed_selection_summary(self):
        self.assertEqual(E1_SEEDS, (42, 64, 128, 256, 512))
        scores = {
            seed: {"2023": float(i), "2024": float(i + 1), "2025": float(i + 2)}
            for i, seed in enumerate(E1_SEEDS)
        }
        summary = e1_seed_summary(scores)
        self.assertEqual(summary["per_seed_cv_official_score"], dict(zip(E1_SEEDS, [1.0, 2.0, 3.0, 4.0, 5.0])))
        self.assertEqual(summary["mean_cv_official_score"], 3.0)
        self.assertAlmostEqual(summary["std_cv_official_score"], np.sqrt(2))
        with self.assertRaisesRegex(ValueError, "exactly these seeds"):
            e1_seed_summary({42: scores[42]})
        with self.assertRaisesRegex(ValueError, "one official score"):
            e1_seed_summary({**scores, 42: {"2023": 1.0}})

    def test_submission_uses_actual_phase1_template(self):
        root = Path(__file__).resolve().parents[1]
        target, _ = load_metadata(root / "data")
        template = root / "sample_submission.csv"
        ids, dates = load_submission_template(template, target)
        self.assertEqual((len(ids), len(dates)), (50, 84))
        self.assertEqual(ids, target)
        predictions = pd.DataFrame(
            [(stock, day, (i + j) % 2) for i, stock in enumerate(ids) for j, day in enumerate(dates)],
            columns=["stock_id", "date", "predicted_y"],
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        submission = build_submission(predictions, template, target)
        self.assertEqual(submission.columns.tolist(), ["id", *dates])
        self.assertEqual(submission["id"].tolist(), ids)
        self.assertEqual(submission.iloc[0, 1:4].tolist(), [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "1 missing"):
            build_submission(predictions.iloc[:-1], template, target)
        extra = pd.concat([
            predictions,
            pd.DataFrame([{"stock_id": ids[0], "date": "2026-10-01", "predicted_y": 1}]),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "1 extra"):
            build_submission(extra, template, target)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_submission(pd.concat([predictions, predictions.iloc[[0]]]), template, target)
        invalid = predictions.copy()
        invalid.loc[0, "predicted_y"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            build_submission(invalid, template, target)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "submission.csv"
            write_submission(predictions, template, output, target)
            written = pd.read_csv(output, dtype={"id": str})
            pd.testing.assert_frame_equal(submission, written)
            with self.assertRaises(FileExistsError):
                write_submission(predictions, template, output, target)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            write_submission(predictions, template, template, target)

    def test_submission_accepts_one_date_session_template(self):
        root = Path(__file__).resolve().parents[1]
        target, _ = load_metadata(root / "data")
        with tempfile.TemporaryDirectory() as temp:
            template = Path(temp) / "session_template.csv"
            with template.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "2026-11-16"])
                writer.writerows((stock, 0) for stock in reversed(target))
            predictions = pd.DataFrame({
                "stock_id": target, "date": ["2026-11-16"] * 50,
                "predicted_y": [1] * 50,
            })
            submission = build_submission(predictions, template, target)
            self.assertEqual(submission["id"].tolist(), list(reversed(target)))
            self.assertEqual(submission.columns.tolist(), ["id", "2026-11-16"])


if __name__ == "__main__":
    unittest.main()
