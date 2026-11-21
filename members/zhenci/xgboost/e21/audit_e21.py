"""Audit January–May 2026 label availability and the existing F0 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e1.save_e1 import _sha256
from shared.data_loader import KEYS, load_global, load_market, load_metadata
from shared.evaluation import CLOSE, OPEN
from shared.features import COL, build_f0_dataset, load_f0_dataset, stock_history_candidates
from shared.preprocessing import F0_FEATURES, NUMERIC_FEATURES
from shared.splits import evaluation_keys
from shared.updated_features import DEFAULT_UPDATE, load_updated_market

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data"
F0 = DATA / "processed/f0.csv"
START, END = "2026-01-01", "2026-05-31"
OUTPUT = Path(__file__).resolve().parent / "audit_outputs"


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    targets, universe = load_metadata(DATA)
    market, global_data = load_market(DATA), load_global(DATA)
    archived = load_f0_dataset(F0)
    recent = market.loc[market["date"].between(START, END)]
    prices = recent[[OPEN, CLOSE]].to_numpy(dtype=float)
    finite = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    if not finite.all() or not np.array_equal(prices[:, 1] > prices[:, 0], recent["y"].to_numpy()):
        raise ValueError("Recent prices or binary labels are invalid")
    if set(recent["stock_id"]) != set(universe):
        raise ValueError("Recent labeled panel does not cover the 100-stock universe")
    recent_f0 = archived.loc[archived["date"].between(START, END)].sort_values(KEYS).reset_index(drop=True)
    if not evaluation_keys(recent).equals(evaluation_keys(recent_f0)):
        raise ValueError("Recent market and F0 keys differ")

    # Removing every observation after May must leave this entire F0 slice unchanged.
    history = market.loc[market["date"].le(END)].drop(columns="y")
    past_global = global_data.loc[global_data["date"].le(END)]
    rebuilt = build_f0_dataset(history, past_global)
    rebuilt = rebuilt.loc[rebuilt["date"].between(START, END)].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(recent_f0, rebuilt, check_exact=False, atol=1e-12, rtol=1e-12)
    sources = stock_history_candidates(history).loc[lambda rows: rows["date"].between(START, END)]
    if not (sources["institutional_source_date"].lt(sources["source_date"])
            & sources["source_date"].lt(sources["date"])).all():
        raise ValueError("Unexpected prior stock or institutional source date")
    # Verify the existing strict-prior global age rule against the source dates of each series.
    series_map = {"vix_normalized": "^VIX", "nasdaq_return": "^IXIC", "sp500_return": "^GSPC", "taiex_return": "^TWII"}
    date_keys = recent_f0[["date"]].drop_duplicates().sort_values("date")
    alignment = []
    for feature, source in series_map.items():
        observed = past_global.loc[past_global[source].notna(), ["date", source]].sort_values("date")
        values = observed[source] / 100 if source == "^VIX" else observed[source] / observed[source].shift() - 1
        sources_aligned = pd.merge_asof(date_keys, observed.assign(value=values).rename(columns={"date":"source_date"}),
            left_on="date", right_on="source_date", allow_exact_matches=False, tolerance=pd.Timedelta(days=5))
        age = (sources_aligned["date"] - sources_aligned["source_date"]).dt.days
        if not age.dropna().between(1, 5).all():
            raise ValueError("Unexpected global age")
        actual = recent_f0.merge(sources_aligned[["date", "value"]], on="date", validate="many_to_one")
        np.testing.assert_allclose(actual[feature], actual["value"], atol=1e-12, rtol=1e-12, equal_nan=True)
        alignment.append(sources_aligned[["date", "source_date"]].assign(feature=feature, age_days=age))

    coverage = recent.groupby("date", as_index=False).agg(stocks=("stock_id", "nunique"), rows=("y", "size"), up=("y", "sum"))
    coverage["absent_from_top100"] = len(universe) - coverage["stocks"]
    by_stock = recent.groupby("stock_id", as_index=False).agg(rows=("y", "size"), first=("date", "min"), last=("date", "max"))
    missing = recent_f0[list(F0_FEATURES)].isna().sum().rename("missing").rename_axis("feature").reset_index()
    missing["missing_rate"] = missing["missing"] / len(recent_f0)
    updated_sources = [DEFAULT_UPDATE, ROOT / "updated_data_17_Nov"]
    inventories = []
    for directory in updated_sources:
        updated, updated_global = load_updated_market(directory, universe), load_global(directory)
        inventories.append({"path": directory.relative_to(ROOT).as_posix(), "rows": len(updated),
            "stock_count": updated["stock_id"].nunique(), "start": updated["date"].min().date().isoformat(),
            "end": updated["date"].max().date().isoformat(), "global_rows": len(updated_global),
            "nonpositive_open_close_rows": int(updated[[OPEN, CLOSE]].le(0).any(axis=1).sum())})
    raw_fields = [COL[key] for key in ("open", "high", "low", "close", "volume", "amount", "trades")]
    base = market.loc[market["date"].le("2025-12-31")]
    scale_checks = []
    for period, frame in (("2020–2025", base), ("2026 January–May", recent)):
        for name in raw_fields:
            column = frame[name].to_numpy(dtype=float)
            scale_checks.append({"period":period,"raw_column":name,"missing_or_nonfinite":int((~np.isfinite(column)).sum()),
                "nonpositive":int((column <= 0).sum()),"p10":np.nanquantile(column,.1),"median":np.nanmedian(column),"p90":np.nanquantile(column,.9)})
    summary = {"recent_start": recent["date"].min().date().isoformat(), "recent_end": recent["date"].max().date().isoformat(),
               "rows": len(recent), "days": len(coverage), "stocks": len(universe), "top50_rows": int(recent["stock_id"].isin(targets).sum()),
               "min_stocks_per_day": int(coverage["stocks"].min()), "max_stocks_per_day": int(coverage["stocks"].max()),
               "absent_stock_dates": int(coverage["absent_from_top100"].sum()), "finite_positive_adjusted_price_rows": int(finite.sum()),
               "label_mismatches": 0, "f0_key_coverage": 1, "numeric_missing": int(recent_f0[list(NUMERIC_FEATURES)].isna().sum().sum()),
               "prefix_rebuild_matches_f0": True,"updates":inventories,
               "input_sha256": {"f0":_sha256(F0),"metadata":_sha256(DATA/"info.json"),
                                "competition_brief":_sha256(ROOT/"final_project.pdf")}}
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("daily_coverage",coverage),("stock_coverage",by_stock),("f0_missingness",missing),
                        ("global_provenance",pd.concat(alignment)),("raw_numeric_checks",pd.DataFrame(scale_checks))):
        frame.to_csv(output/f"{name}.csv",index=False)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
