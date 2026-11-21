"""Fixed E18 catalogue and feature construction from completed stock sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shared.data_loader import KEYS, validate_keys
from shared.features import COL
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES

HERE = Path(__file__).resolve().parent
CATALOGUE = HERE / "feature_catalogue.json"
RAW = tuple(COL[k] for k in ("open", "high", "low", "close", "volume", "amount", "trades"))


def catalogue() -> list[dict]:
    entries = []

    def add(family, name, formula, inputs, lookback, overlap="none", included=True, scope="stock"):
        lag = {"stock": "Latest same-stock session strictly before prediction date",
               "market": "Latest supplied-universe session strictly before prediction date",
               "relative": "Latest same-stock session strictly before prediction date, peers on that source date"}
        entries.append({"name": name, "family": family, "formula": formula,
                        "raw_columns": [COL[field] for field in inputs.split()],
                        "lookback_sessions": lookback, "scope": scope, "availability": lag[scope],
                        "supported": True, "included": included, "overlap": overlap,
                        "origin": "Independent adaptation of requested feature family"})

    for name, formula in (("candle_body", "(C-O)/(H-L)"), ("upper_wick", "(H-max(O,C))/(H-L)"),
                          ("lower_wick", "(min(O,C)-L)/(H-L)"), ("close_location", "(C-L)/(H-L)"),
                          ("open_location", "(O-L)/(H-L)")):
        add("A", name, formula, "open high low close", 1)
    add("A", "overnight_gap_abs", "abs(q_s)", "open close", 2, "Absolute value of E2's signed gap")
    for n in (3, 10, 20, 60):
        add("A", f"momentum_{n}", f"C_s/C_(s-{n})-1", "close", n + 1,
            "Exact E2 F1 momentum_10d" if n == 10 else "none")
    for name, r in (("intraday", "r"), ("overnight", "q")):
        for n in (3, 5, 10, 20):
            add("A", f"{name}_compound_{n}", f"product of (1+{r}) over {n} sessions minus 1",
                "open close", n + int(name == "overnight"), "E4 uses arithmetic means on 5/20 sessions")
    for n in (5, 10, 20):
        add("A", f"intraday_positive_fraction_{n}", f"mean(1[r>0], {n})", "open close", n)
    for n in (5, 10, 20, 60):
        add("B", f"ma_distance_{n}", f"C/mean(C,{n})-1", "close", n, "F0 uses supplied SMA of unknown settings")
    for n in (10, 20, 60):
        for endpoint in ("max", "min"):
            add("B", f"rolling_{endpoint}_distance_{n}", f"C/{endpoint}(C,{n})-1", "close", n)
    for n in (10, 20):
        add("B", f"log_trend_slope_{n}", "OLS slope of log(C) on equal session steps", "close", n)
        add("B", f"log_trend_r2_{n}", "Squared correlation of log(C) with equal session steps", "close", n)
    for name, r, raw in (("intraday", "r", "open close"), ("close", "u", "close")):
        for n in (5, 10, 20):
            add("B", f"{name}_volatility_{n}", f"population std({r},{n})", raw, n + int(name == "close"),
                "Exact E2 F1 close_volatility_5d" if name == "close" and n == 5 else "none")
    for n in (10, 20):
        add("B", f"downside_semideviation_{n}", f"sqrt(mean(min(u,0)^2,{n}))", "close", n + 1)
    for n in (5, 10, 20, 30):
        add("B", f"mean_normalized_range_{n}", f"mean((H-L)/C,{n})", "high low close", n,
            "Rolling mean of F0 price_range_1d")
    for n in (5, 10, 20):
        add("B", f"normalized_atr_{n}", f"mean(TR,{n})/C_s", "high low close", n + 1,
            "F0 uses supplied ATR of unknown settings")
    for short, long in ((5, 20), (10, 60)):
        add("B", f"volatility_ratio_{short}_{long}", f"std(u,{short})/std(u,{long})", "close", long + 1)
    for n in (5, 10, 60):
        add("C", f"relative_volume_{n}", f"V/mean(V,{n})", "volume", n, "F0 uses a 20-session window")
    for n in (5, 20):
        add("C", f"relative_amount_{n}", f"A/mean(A,{n})", "amount", n,
            "Exact F0 relative_amount_20d" if n == 20 else "Different window from F0", included=n != 20)
    for n in (10, 20):
        add("C", f"volume_cv_{n}", f"std(V,{n})/mean(V,{n})", "volume", n)
        add("C", f"return_volume_corr_{n}", f"Pearson corr(u,V) over {n} sessions", "close volume", n + 1)
        add("C", f"amihud_{n}", f"mean(abs(u)/A,{n}) per reported traded-amount unit", "close amount", n + 1)
    add("C", "amount_per_transaction", "A/T", "amount trades", 1, "F0 trade size uses shares per transaction")
    for name, formula, inputs, window in (
        ("market_breadth", "mean across stocks of 1[r>0]", "open close", 1),
        ("market_intraday_mean", "cross-sectional mean(r)", "open close", 1),
        ("market_intraday_median", "cross-sectional median(r)", "open close", 1),
        ("market_intraday_dispersion", "cross-sectional population std(r)", "open close", 1),
        ("market_overnight_median", "cross-sectional median(q)", "open close", 2),
        ("market_relative_volume_median", "cross-sectional median(V/mean(V,20))", "volume", 20),
    ):
        add("D", name, formula, inputs, window, scope="market")
    for n in (5, 20):
        add("D", f"market_breadth_{n}", f"mean(daily market breadth,{n})", "open close", n, scope="market")
        add("D", f"market_volatility_{n}", f"population std(daily equal-weight intraday return,{n})",
            "open close", n, scope="market")
    for name, formula, inputs, n, overlap in (
        ("intraday_vs_peer_median", "r - cross-sectional median(r)", "open close", 1, "E3 relative feature agrees when all peers share the same latest session"),
        ("overnight_vs_peer_median", "q - cross-sectional median(q)", "open close", 2, "none"),
        *((f"momentum_vs_peer_median_{n}", f"m_{n} - cross-sectional median(m_{n})", "close", n + 1,
           "E3 relative feature agrees on aligned sessions for n=5" if n == 5 else "none") for n in (5, 20)),
        *((f"volatility_vs_peer_median_{n}", f"std(u,{n}) - cross-sectional median(std(u,{n}))", "close", n+1, "none") for n in (5, 20)),
    ):
        add("D", name, formula, inputs, n, overlap, scope="relative")
    return entries


FAMILIES = {family: tuple(row["name"] for row in catalogue() if row["family"] == family and row["included"])
            for family in "ABCD"}
EXTRAS = tuple(feature for fields in FAMILIES.values() for feature in fields)
MARKET_FIELDS = tuple(row["name"] for row in catalogue() if row["scope"] == "market")


def rolling(series: pd.Series, n: int):
    return series.rolling(n, min_periods=n)


def divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b.ne(0))).replace([np.inf, -np.inf], np.nan)


def stock_windows(rows: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v, a, t = (rows[key].where(np.isfinite(rows[key]) & rows[key].gt(0)) for key in RAW)
    valid_candle = h.ge(pd.concat([o, c], axis=1).max(axis=1, skipna=False)) & l.le(pd.concat([o, c], axis=1).min(axis=1, skipna=False))
    r, q, u = divide(c, o) - 1, divide(o, c.shift()) - 1, divide(c, c.shift()) - 1
    span = (h-l).where(valid_candle)
    gain = r.gt(0).astype(float).where(r.notna())
    result = {"candle_body": divide(c-o, span), "upper_wick": divide(h-pd.concat([o,c], axis=1).max(axis=1), span),
              "lower_wick": divide(pd.concat([o,c], axis=1).min(axis=1)-l, span),
              "close_location": divide(c-l, span), "open_location": divide(o-l, span),
              "overnight_gap_abs": q.abs(), "_r": r, "_q": q, "_gain": gain,
              "_relative_volume20": divide(v, rolling(v,20).mean())}
    for n in (3, 5, 10, 20, 60):
        result["_momentum5" if n == 5 else f"momentum_{n}"] = divide(c,c.shift(n))-1
    for label, series in (("intraday", r), ("overnight", q)):
        for n in (3, 5, 10, 20):
            result[f"{label}_compound_{n}"] = np.expm1(rolling(np.log1p(series), n).sum())
    for n in (5, 10, 20):
        result[f"intraday_positive_fraction_{n}"] = rolling(gain,n).mean()
        result[f"intraday_volatility_{n}"] = rolling(r,n).std(ddof=0)
        result[f"close_volatility_{n}"] = rolling(u,n).std(ddof=0)
    for n in (5, 10, 20, 60):
        result[f"ma_distance_{n}"] = divide(c,rolling(c,n).mean())-1
    for n in (10, 20, 60):
        result[f"rolling_max_distance_{n}"] = divide(c,rolling(c,n).max())-1
        result[f"rolling_min_distance_{n}"] = divide(c,rolling(c,n).min())-1
    log_c = np.log(c)
    for n in (10, 20):
        centered = np.arange(n) - (n-1)/2
        slope = rolling(log_c,n).apply(lambda y: float(np.dot(centered,y-y.mean()) / np.dot(centered,centered)), raw=True)
        sigma = rolling(log_c,n).std(ddof=0)
        result[f"log_trend_slope_{n}"] = slope
        result[f"log_trend_r2_{n}"] = divide(slope * centered.std(), sigma).pow(2).clip(upper=1)
        result[f"downside_semideviation_{n}"] = np.sqrt(rolling(u.clip(upper=0).pow(2),n).mean())
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1,skipna=False)
    for n in (5, 10, 20, 30):
        result[f"mean_normalized_range_{n}"] = rolling(divide(h-l,c),n).mean()
        if n != 30:
            result[f"normalized_atr_{n}"] = divide(rolling(tr,n).mean(),c)
    for short,long in ((5,20),(10,60)):
        result[f"volatility_ratio_{short}_{long}"] = divide(result[f"close_volatility_{short}"],rolling(u,long).std(ddof=0))
    for n in (5,10,60):
        result[f"relative_volume_{n}"] = divide(v,rolling(v,n).mean())
    result["relative_amount_5"] = divide(a,rolling(a,5).mean())
    result["amount_per_transaction"] = divide(a,t)
    for n in (10,20):
        result[f"volume_cv_{n}"] = divide(rolling(v,n).std(ddof=0),rolling(v,n).mean())
        corr = rolling(u,n).corr(v)
        result[f"return_volume_corr_{n}"] = corr.where(
            np.isfinite(corr) & rolling(u,n).std(ddof=0).gt(0) & rolling(v,n).std(ddof=0).gt(0)).clip(-1,1)
        result[f"amihud_{n}"] = rolling(divide(u.abs(),a),n).mean()
    return rows[KEYS].join(pd.DataFrame(result,index=rows.index)).assign(stock_source_session=np.arange(1,len(rows)+1))


def build_features(market: pd.DataFrame, keys: pd.DataFrame, reference_ids: list[str]
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build at requested keys with strict-as-of stock and market source dates."""
    validate_keys(market)
    validate_keys(keys)
    if len(reference_ids) != 100 or set(market["stock_id"]) != set(reference_ids) or not set(keys["stock_id"]) <= set(reference_ids):
        raise ValueError("E18 requires the full metadata Top100 supply history")
    raw = market.loc[:, KEYS + list(RAW)].sort_values(["stock_id","date"])
    sessions = pd.concat([stock_windows(group) for _,group in raw.groupby("stock_id",sort=False)],ignore_index=True)
    by_date = sessions.groupby("date",sort=True)
    aggregates = by_date.agg(market_breadth=("_gain","mean"), market_intraday_mean=("_r","mean"),
                            market_intraday_median=("_r","median"),
                            market_intraday_dispersion=("_r",lambda x: x.std(ddof=0)),
                            market_overnight_median=("_q","median"),
                            market_relative_volume_median=("_relative_volume20","median"))
    for n in (5,20):
        aggregates[f"market_breadth_{n}"] = rolling(aggregates["market_breadth"],n).mean()
        aggregates[f"market_volatility_{n}"] = rolling(aggregates["market_intraday_mean"],n).std(ddof=0)
    relative = {"intraday_vs_peer_median":"_r", "overnight_vs_peer_median":"_q", "momentum_vs_peer_median_5":"_momentum5",
                "momentum_vs_peer_median_20":"momentum_20", "volatility_vs_peer_median_5":"close_volatility_5",
                "volatility_vs_peer_median_20":"close_volatility_20"}
    for name, column in relative.items():
        sessions[name] = sessions[column] - by_date[column].transform("median")
    coverage = by_date[list(dict.fromkeys([*relative.values(),"_relative_volume20"]))].count().add_prefix("valid_")
    coverage["present_stocks"] = by_date.size()
    coverage["reference_stocks"] = 100
    coverage["market_source_session"] = np.arange(1,len(coverage)+1)
    source_keys = sessions[KEYS + ["stock_source_session"]].copy()
    stock = sessions[[*KEYS,"stock_source_session",*(name for name in EXTRAS if name not in MARKET_FIELDS)]]
    stock = stock.rename(columns={"date":"stock_source_date"}).sort_values("stock_source_date")
    context = aggregates.assign(market_source_session=coverage["market_source_session"]).rename_axis("market_source_date").reset_index()
    out = pd.merge_asof(keys[KEYS].sort_values("date"),stock,by="stock_id",left_on="date",right_on="stock_source_date",allow_exact_matches=False)
    out = pd.merge_asof(out,context,left_on="date",right_on="market_source_date",allow_exact_matches=False)
    for name in ("stock_source_date","market_source_date"):
        if (out[name].notna() & out[name].ge(out["date"])).any():
            raise ValueError(f"Invalid source-date alignment: {name}")
    ordered = [*KEYS,*EXTRAS,"stock_source_date","stock_source_session","market_source_date","market_source_session"]
    return out[ordered].sort_values(["date","stock_id"]).reset_index(drop=True), coverage.reset_index(), source_keys


class ExpandedPreprocessor:
    """Apply shared F0 transformations and the same numeric policy to extras."""

    def __init__(self, extras: tuple[str,...]):
        self.extras = extras

    def fit(self, train: pd.DataFrame):
        self.base = BaselinePreprocessor().fit(train[list(F0_FEATURES)])
        extra = train[list(self.extras)].replace([np.inf,-np.inf],np.nan)
        self.median = extra.median()
        if self.median.isna().any():
            raise ValueError(f"Completely missing training feature: {self.median[self.median.isna()].index.tolist()}")
        filled = extra.fillna(self.median)
        self.mean, self.scale = filled.mean(), filled.std(ddof=0).replace(0,1)
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        baseline = self.base.transform(rows[list(F0_FEATURES)])
        extra = rows[list(self.extras)].replace([np.inf,-np.inf],np.nan)
        return pd.concat([baseline,(extra.fillna(self.median)-self.mean)/self.scale],axis=1)

    def state(self) -> dict:
        return {"base":self.base.to_state(),"extras":list(self.extras),"median":self.median.to_dict(),
                "mean":self.mean.to_dict(),"scale":self.scale.to_dict()}

    @classmethod
    def from_state(cls,state):
        obj = cls(tuple(state["extras"]))
        obj.base = BaselinePreprocessor.from_state(state["base"])
        for field in ("median","mean","scale"):
            setattr(obj,field,pd.Series(state[field],dtype=float).reindex(obj.extras))
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-catalogue",action="store_true",required=True)
    parser.parse_args()
    with CATALOGUE.open("x",encoding="utf-8") as file:
        json.dump(catalogue(),file,indent=2,ensure_ascii=False)
        file.write("\n")
