"""Compare the fixed E3 recipe at two training cutoffs on June–September 2023–2025."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e1.save_e1 import _sha256
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY,
    align_saved_probabilities, load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from members.zhenci.xgboost.e15.train_e15 import BLOCK_DAYS, DRAWS, interval
from members.zhenci.xgboost.e17.train_e17 import fit_classifier
from members.zhenci.xgboost.e19.diagnose_e19 import daily_stats, distribution
from members.zhenci.xgboost.e21.audit_e21 import DATA, F0, ROOT
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import OPEN, CLOSE, cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, SEED, Split, split_market

OUTPUT = Path(__file__).resolve().parent / "outputs"


def paired_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="score_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts["static"].equals(counts["updated"]):
        raise ValueError("Paired scores require identical trading-date coverage")
    rng = np.random.default_rng(SEED)
    rows, points, samples = [], [], []
    for fold in FOLDS:
        delta = (sums.loc[fold.name, "updated"] - sums.loc[fold.name, "static"]).to_numpy()
        n = counts.loc[fold.name, "static"].to_numpy()
        days = len(n)
        starts = rng.integers(0, days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
        index = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days).reshape(DRAWS, -1)[:, :days]
        point = float(delta.sum() / n.sum())
        draws = delta[index].sum(axis=1) / n[index].sum(axis=1)
        points.append(point)
        samples.append(draws)
        rows.append(interval("updated", "static", fold.name, point, draws))
    rows.append(interval("updated", "static", "equal_year", np.mean(points), np.mean(samples, axis=0)))
    return pd.DataFrame(rows)


def score_days(rows: pd.DataFrame) -> pd.DataFrame:
    daily = daily_stats(rows)
    signed = rows.assign(position=2 * rows["predicted_y"] - 1).groupby("date", as_index=False).agg(
        mean_position=("position", "mean"), mean_return=("realized_return", "mean"))
    daily = daily.merge(signed, on="date", validate="one_to_one")
    daily["market_direction"] = daily["mean_position"] * daily["mean_return"]
    daily["stock_selection"] = daily["daily_score"] - daily["market_direction"]
    daily["market_sum"] = daily["market_direction"] * daily["n"]
    daily["selection_sum"] = daily["stock_selection"] * daily["n"]
    return daily


def period_results(daily: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    result = daily.groupby(groups, as_index=False).agg(
        n=("n", "sum"), trading_days=("date", "size"), score_sum=("score_sum", "sum"),
        market_sum=("market_sum", "sum"), selection_sum=("selection_sum", "sum"),
        mean_daily_spearman=("spearman", "mean"), daily_score_mean=("daily_score", "mean"),
        daily_score_sd=("daily_score", "std"), daily_p10=("daily_score", lambda v: v.quantile(.1)),
        daily_median=("daily_score", "median"), daily_p90=("daily_score", lambda v: v.quantile(.9)),
        negative_day_rate=("daily_score", lambda v: v.lt(0).mean()))
    for name, summed in (("official_score", "score_sum"), ("market_direction", "market_sum"),
                         ("stock_selection", "selection_sum")):
        result[name] = result.pop(summed) / result["n"]
    return result


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    archive_recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3 = load_fixed_rule(DEFAULT_E3_SUMMARY, archive_recipe["config_id"])
    if archive_recipe["num_boost_round"] != 100:
        raise ValueError("E21 requires the original fixed 100-round recipe")
    expected = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    archive = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, archive_recipe)
    target_ids, all_ids = load_metadata(DATA)
    end = FOLDS[-1].eval_end
    market = load_market(DATA).loc[lambda rows: rows["date"].le(end)]
    f0 = load_f0_dataset(F0).loc[lambda rows: rows["date"].le(end)]
    metrics, daily, predictions, runtime = [], [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="e21-", dir=output.parent) as temporary:
        out = Path(temporary) / "outputs"
        out.mkdir()
        for fold in FOLDS:
            year = fold.name
            start, stop = f"{year}-06-01", f"{year}-09-30"
            cutoff = f"{year}-05-31"
            static_train, full_valid = split_market(market, target_ids, fold)
            recent_train, valid = split_market(market, target_ids, Split(year, cutoff, start, stop))
            both_same_keys = full_valid.loc[full_valid["date"].between(start, stop), KEYS].reset_index(drop=True)
            if (not valid[KEYS].equals(both_same_keys)
                    or not recent_train.iloc[:len(static_train)][KEYS + ["y"]].equals(static_train[KEYS + ["y"]])
                    or set(static_train["stock_id"]) != set(all_ids)
                    or set(recent_train["stock_id"]) != set(all_ids)
                    or recent_train["date"].max() > pd.Timestamp(cutoff) or valid["date"].min() <= pd.Timestamp(cutoff)):
                raise ValueError(f"Cutoff, universe or stock-date support changed in {year}")
            extra = recent_train.iloc[len(static_train):]
            if extra.empty or not extra["date"].gt(fold.train_end).all():
                raise ValueError(f"The added rows must be strictly after the static cutoff in {year}")
            static_p = align_saved_probabilities(full_valid, archive, year, SEED)
            prior_tai = join_f0(full_valid, f0, include_label=False)["taiex_return"].to_numpy()
            control_calls = regime_predictions(static_p, prior_tai, rule)
            prior_score = evaluate_predictions(full_valid, full_valid[KEYS].assign(predicted_y=control_calls))["official_score"]
            np.testing.assert_allclose(prior_score, expected["fold_official_scores_by_seed"][str(SEED)][year], atol=1e-12, rtol=0)
            paired_static = valid[KEYS].merge(full_valid[KEYS].assign(p_up=static_p), on=KEYS, validate="one_to_one")
            if not paired_static[KEYS].equals(valid[KEYS]):
                raise ValueError(f"Static alignment changed for {year}")

            began = perf_counter()
            train_f0, valid_f0 = (join_f0(rows, f0, include_label=False) for rows in (recent_train, valid))
            pre = BaselinePreprocessor().fit(train_f0[list(F0_FEATURES)])
            x_train, x_valid = (pre.transform(rows[list(F0_FEATURES)]) for rows in (train_f0, valid_f0))
            prepared = perf_counter()
            fitted = fit_classifier(archive_recipe, x_train, recent_train, pd.Series(1.0, index=recent_train.index))
            trained = perf_counter()
            updated_p = fitted.predict_proba(x_valid)[:, 1].astype(float)
            runtime.append({"fold":year, "static_cutoff":fold.train_end, "updated_cutoff":cutoff,
                            "static_rows":len(static_train), "updated_rows":len(recent_train), "added_rows":len(extra),
                            "static_stocks":static_train["stock_id"].nunique(), "updated_stocks":recent_train["stock_id"].nunique(),
                            "added_days":extra["date"].nunique(), "added_stocks":extra["stock_id"].nunique(),
                            "preprocessing_seconds":prepared-began,"updated_fit_seconds":trained-prepared,
                            "updated_inference_seconds":perf_counter()-trained})
            labels = valid["y"].to_numpy()
            realized = ((valid[CLOSE] - valid[OPEN]) / valid[OPEN]).to_numpy()
            if not np.isfinite(realized).all() or not np.array_equal(realized > 0, labels):
                raise ValueError("Validation labels differ from intraday direction")
            tai = valid_f0["taiex_return"].to_numpy()
            for arm, probability in (("static", paired_static["p_up"].to_numpy()), ("updated", updated_p)):
                calls = regime_predictions(probability, tai, rule)
                frame = valid[KEYS].assign(predicted_y=calls, p_up=probability, actual_up=labels,
                                          realized_return=realized, contribution=(2*calls-1)*realized)
                official = evaluate_predictions(valid, frame[KEYS + ["predicted_y"]])
                metrics.append({"fold":year, "arm":arm, **official, **distribution("validation", labels, probability, calls)})
                daily.append(score_days(frame).assign(fold=year, arm=arm))
                predictions.append(frame.assign(fold=year, arm=arm))
                print(f"{year} {arm}: {official['official_score']:+.8f}", flush=True)
            model = fitted.get_booster()
            if model.num_boosted_rounds() != 100:
                raise ValueError("Exactly 100 fitted boosting rounds are required")
            folder = out / "models" / year
            folder.mkdir(parents=True)
            model.save_model(str(folder / "model.json"))
            (folder / "preprocessor.json").write_text(json.dumps(pre.to_state(), indent=2, allow_nan=False) + "\n",encoding="utf-8")
            restored = xgb.Booster(model_file=str(folder / "model.json"))
            np.testing.assert_allclose(restored.predict(xgb.DMatrix(x_valid)), updated_p, rtol=0, atol=1e-7)
            (folder / "manifest.json").write_text(json.dumps({"train_start":"2020-01-01", "train_end":cutoff,
                "training_rows":len(recent_train), "training_stocks":len(all_ids), "seed":SEED, "rounds":100,
                "params":archive_recipe["params"], "taiex_rule":rule, "feature_order":list(F0_FEATURES),
                "file_sha256":{name:_sha256(folder/name) for name in ("model.json","preprocessor.json")}}, indent=2)+"\n",encoding="utf-8")

        metric_table = pd.DataFrame(metrics)
        daily_table = pd.concat(daily, ignore_index=True)
        daily_table["month"] = daily_table["date"].dt.strftime("%Y-%m")
        period = period_results(daily_table, ["fold", "arm"])
        monthly = period_results(daily_table, ["fold", "arm", "month"])
        checked = metric_table.merge(period, on=["fold","arm"], suffixes=("", "_daily"), validate="one_to_one")
        np.testing.assert_allclose(checked["official_score"], checked["official_score_daily"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(checked["official_score"], checked["market_direction"] + checked["stock_selection"], atol=1e-12, rtol=0)
        interval_table = paired_intervals(daily_table)
        scores = {arm: cv_score(dict(zip(group["fold"],group["official_score"]))) for arm,group in metric_table.groupby("arm")}
        summary = {"equal_year_official_scores":scores,"worst_period_scores":period.groupby("arm")["official_score"].min().to_dict(),
                   "seed":SEED,"params":archive_recipe["params"],"rounds":100,"taiex_rule":rule,
                   "bootstrap_draws":DRAWS,"bootstrap_days":BLOCK_DAYS,
                   "versions":{"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"xgboost":xgb.__version__},
                   "inputs_sha256":{path.relative_to(ROOT).as_posix():_sha256(path) for path in
                       (F0, DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY, DATA/"info.json", Path(__file__))},
                   "original_market_sha256":{stock:_sha256(DATA/"market"/f"{stock}.csv") for stock in all_ids}}
        datasets = {"period_metrics":checked.drop(columns="official_score_daily"), "daily_scores":daily_table,
                    "monthly_scores":monthly,"validation_predictions":pd.concat(predictions,ignore_index=True),
                    "paired_uncertainty":interval_table,"training_counts_runtime":pd.DataFrame(runtime)}
        for name,table in datasets.items():
            table.to_csv(out/f"{name}.csv",index=False)
        (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        out.rename(output)
    print(json.dumps(scores,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
