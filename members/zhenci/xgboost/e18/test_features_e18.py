"""Verify alignment and leakage resistance with synthetic stock sessions."""

import unittest

import numpy as np
import pandas as pd

from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e18.features_e18 import EXTRAS, RAW, build_features, stock_windows
from shared.data_loader import KEYS
from shared.features import COL


class FeatureAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = [f"{n:04}" for n in range(100)]
        cls.days = pd.bdate_range("2022-01-03", periods=70)
        raw = pd.MultiIndex.from_product([cls.ids, cls.days], names=KEYS).to_frame(index=False)
        for stock, group in raw.groupby("stock_id"):
            n = np.arange(len(group), dtype=float)
            o = 100 + n
            c = o + 1 + int(stock) / 100
            raw.loc[group.index, list(RAW)] = np.column_stack((o,c+2,o-2,c,1000+n*3,10000+n*50,10+n))
        cls.raw = raw.drop(raw.index[(raw.stock_id == cls.ids[0]) & (raw.date == cls.days[-2])])

    def test_future_changes_and_new_key(self):
        keys = pd.DataFrame({"stock_id":self.ids[:2],"date":self.days[-1]})
        baseline, coverage, _ = build_features(self.raw,keys,self.ids)
        changed = self.raw.copy()
        changed.loc[changed.date.ge(self.days[-1]),list(RAW)] *= 10
        future = changed.loc[changed.date.eq(self.days[-1])].copy()
        future["date"] += pd.Timedelta(days=1)
        changed = pd.concat([changed,future],ignore_index=True)
        expanded, _, _ = build_features(changed,keys,self.ids)
        pd.testing.assert_frame_equal(baseline,expanded)
        self.assertEqual(baseline.loc[0,"stock_source_date"],self.days[-3])
        self.assertEqual(baseline.loc[1,"stock_source_date"],self.days[-2])
        self.assertEqual(baseline.loc[0,"market_source_date"],self.days[-2])
        self.assertEqual(coverage.set_index("date").loc[self.days[-2],"present_stocks"],99)
        hypothetical = keys.assign(date=self.days[-1]+pd.Timedelta(days=1))
        forecast,_,_ = build_features(self.raw,hypothetical,self.ids)
        self.assertTrue(forecast["stock_source_date"].eq(self.days[-1]).all())

    def test_roll_ends_at_source_session(self):
        keys = pd.DataFrame({"stock_id":[self.ids[1]],"date":[self.days[-1]]})
        features,_,_ = build_features(self.raw,keys,self.ids)
        stock = self.raw.loc[self.raw.stock_id.eq(self.ids[1]) & self.raw.date.lt(self.days[-1])]
        c,o = stock[COL["close"]].to_numpy(),stock[COL["open"]].to_numpy()
        self.assertAlmostEqual(features.loc[0,"momentum_60"],c[-1]/c[-61]-1)
        self.assertAlmostEqual(features.loc[0,"intraday_compound_5"],np.prod(c[-5:]/o[-5:])-1)
        self.assertAlmostEqual(features.loc[0,"overnight_compound_3"],np.prod(o[-3:]/c[-4:-1])-1)
        self.assertEqual(set(features.columns)-set(KEYS)-set(EXTRAS),
                         {"stock_source_date","stock_source_session","market_source_date","market_source_session"})

    def test_invalid_divisions_and_full_warmup(self):
        stock = self.raw.loc[self.raw.stock_id.eq(self.ids[1])].copy()
        stock.loc[stock.index[-1],COL["trades"]] = 0
        stock.loc[stock.index[0],list(RAW[:4])] = 100
        result=stock_windows(stock)
        self.assertTrue(np.isnan(result["candle_body"].iloc[0]))
        self.assertTrue(np.isnan(result["amount_per_transaction"].iloc[-1]))
        self.assertTrue(result["momentum_60"].iloc[:60].isna().all())
        self.assertTrue(np.isfinite(result["momentum_60"].iloc[60]))


if __name__ == "__main__":
    unittest.main()
