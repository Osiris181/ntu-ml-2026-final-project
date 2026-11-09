"""Read-only comparison of prior-session intraday direction, E1, and E3.

Run from the repository root:
    uv run python -m members.zhenci.diagnostics.previous_day_gap

The candidate rule is NOT identified as the leaderboard's "Previous day"
baseline. This script reads saved predictions; it never fits a model or rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import evaluate_predictions
from shared.features import load_f0_dataset, stock_history_candidates
from shared.splits import E1_SEEDS, FOLDS, HOLDOUT, split_market
from shared.submission import load_submission_template
from shared.updated_features import load_updated_market

OUTPUT_DIR = Path(__file__).resolve().parent / "previous_day_gap_outputs"
NAMES = ("candidate", "e1", "e3")


def _submission(path: Path, ids: list[str], dates: list[str]) -> pd.DataFrame:
    wide = pd.read_csv(path, dtype={"id": str})
    if wide.columns.tolist() != ["id", *dates] or wide["id"].tolist() != ids:
        raise ValueError(f"Unexpected submission keys: {path}")
    values = wide[dates]
    if values.isna().any().any() or not values.isin([0, 1]).all().all():
        raise ValueError(f"Nonbinary or missing submission values: {path}")
    long = wide.melt(id_vars="id", var_name="date", value_name="predicted_y")
    long = long.rename(columns={"id": "stock_id"})
    long["date"] = pd.to_datetime(long["date"])
    validate_keys(long)
    return long


def _same_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
    validate_keys(left)
    validate_keys(right)
    a = pd.MultiIndex.from_frame(left[KEYS])
    b = pd.MultiIndex.from_frame(right[KEYS])
    if len(a) != len(b) or not a.difference(b).empty or not b.difference(a).empty:
        raise ValueError("Evaluation and prediction keys differ")


def _period_rows(evaluate: pd.DataFrame, predictions: pd.DataFrame,
                 f0: pd.DataFrame, history: pd.DataFrame, period: str) -> pd.DataFrame:
    _same_keys(evaluate, predictions)
    if predictions[["e1", "e3"]].isna().any().any():
        raise ValueError(f"Missing E1/E3 prediction: {period}")
    if not predictions[["e1", "e3"]].isin([0, 1]).all().all():
        raise ValueError(f"Nonbinary E1/E3 prediction: {period}")
    joined = evaluate[KEYS + ["etl:adj_open", "etl:adj_close"]].merge(
        predictions[KEYS + ["e1", "e3"]], on=KEYS, validate="one_to_one"
    ).merge(f0[KEYS + ["intraday_return_1d", "daily_return_1d", "taiex_return"]],
            on=KEYS, validate="one_to_one", how="left", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Missing F0 keys: {period}")
    joined = joined.drop(columns="_merge")
    joined = joined.merge(history[KEYS + ["source_date", "intraday_return_1d"]],
                          on=KEYS, validate="one_to_one", suffixes=("", "_source"))
    if (joined["source_date"].isna().any()
            or not joined["source_date"].lt(joined["date"]).all()):
        raise ValueError(f"Candidate prior-session date invalid: {period}")
    if not np.allclose(joined["intraday_return_1d"], joined["intraday_return_1d_source"],
                       rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError(f"F0 prior return disagrees with stock history: {period}")
    if joined["intraday_return_1d"].isna().any():
        raise ValueError(f"Candidate has missing prior return: {period}")
    joined["candidate"] = joined["intraday_return_1d"].gt(0).astype(int)
    joined["realized_return"] = ((joined["etl:adj_close"] - joined["etl:adj_open"])
                                  / joined["etl:adj_open"])
    joined["truth"] = joined["realized_return"].gt(0).astype(int)
    for name in NAMES:
        joined[name] = joined[name].astype(int)
        joined[f"{name}_contribution"] = (2 * joined[name] - 1) * joined["realized_return"]
        official = evaluate_predictions(
            evaluate, joined[KEYS + [name]].rename(columns={name: "predicted_y"})
        )
        if not np.isclose(official["official_score"], joined[f"{name}_contribution"].mean(),
                          rtol=0, atol=1e-14):
            raise ValueError(f"Score decomposition mismatch: {period}/{name}")
    joined.insert(0, "period", period)
    return joined.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _disagreement(rows: pd.DataFrame, model: str, baseline: str = "candidate") -> dict:
    changed = rows.loc[rows[model].ne(rows[baseline])].copy()
    delta = changed[f"{model}_contribution"] - changed[f"{baseline}_contribution"]
    harmful = -delta.loc[delta < 0]
    beneficial = delta.loc[delta > 0]
    return {
        "period": rows["period"].iat[0], "model": model, "reference": baseline,
        "n_disagreements": len(changed),
        "reference_correct_model_wrong": int((changed[baseline].eq(changed["truth"])
                                              & changed[model].ne(changed["truth"])).sum()),
        "model_correct_reference_wrong": int((changed[model].eq(changed["truth"])
                                              & changed[baseline].ne(changed["truth"])).sum()),
        "positive_delta_rows": int(delta.gt(0).sum()),
        "negative_delta_rows": int(delta.lt(0).sum()),
        "zero_delta_rows": int(delta.eq(0).sum()),
        "benefit_sum": float(beneficial.sum()),
        "harm_sum": float(harmful.sum()),
        "score_delta_on_all_rows": float(delta.sum() / len(rows)),
        "top_10_harms_fraction": float(harmful.nlargest(10).sum() / harmful.sum()) if harmful.sum() else 0.0,
        "top_50_harms_fraction": float(harmful.nlargest(50).sum() / harmful.sum()) if harmful.sum() else 0.0,
    }


def run(output_dir: Path = OUTPUT_DIR) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    data_dir = PROJECT_ROOT / "data"
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    updated = load_updated_market(PROJECT_ROOT / "updated_data_7_Nov/data", all_ids)
    if updated["date"].min() <= market["date"].max():
        raise ValueError("Updated market overlaps the original history")
    full_history = pd.concat([market.drop(columns="y"), updated], ignore_index=True)
    history = stock_history_candidates(full_history)
    f0 = load_f0_dataset(data_dir / "processed/f0.csv")
    f0_updated = load_f0_dataset(data_dir / "processed/f0_with_updates.csv")
    with (PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json").open(
        encoding="utf-8"
    ) as file:
        rule_summary = json.load(file)
    source = pd.read_csv(PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv",
                         dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    if set(source["seed"]) != set(E1_SEEDS) or set(source["fold"]) != {f.name for f in FOLDS}:
        raise ValueError("Incomplete saved E1 CV predictions")
    if source["config_id"].nunique() != 1:
        raise ValueError("Mixed E1 configurations")
    if source.duplicated(["fold", "seed", *KEYS]).any():
        raise ValueError("Duplicate E1 CV keys")
    if source.groupby(["fold", *KEYS])["p_up"].nunique().ne(1).any():
        raise ValueError("Five E1 seeds differ; handle each seed separately")
    rule = selected_rule(rule_summary, source["config_id"].iat[0])
    source = source.loc[source["seed"].eq(42)].copy()
    if not source["predicted_y"].eq(source["p_up"].gt(0.5).astype(int)).all():
        raise ValueError("Saved E1 labels do not match probabilities")

    periods = []
    for fold in FOLDS:
        train, evaluate = split_market(market, target_ids, fold)
        if train["date"].max() >= evaluate["date"].min():
            raise ValueError("Training/evaluation overlap")
        saved = source.loc[source["fold"].eq(fold.name)].merge(
            f0[KEYS + ["taiex_return"]], on=KEYS, validate="one_to_one"
        )
        pred = saved[KEYS].copy()
        pred["e1"] = saved["predicted_y"].to_numpy()
        pred["e3"] = regime_predictions(saved["p_up"].to_numpy(),
                                         saved["taiex_return"].to_numpy(), rule)
        periods.append(_period_rows(evaluate, pred, f0, history, fold.name))

    _, holdout_eval = split_market(market, target_ids, HOLDOUT)
    holdout = pd.read_csv(PROJECT_ROOT / "results/zhenci_xgboost_e7_holdout_v1/holdout_predictions.csv",
                          dtype={"stock_id": str}, parse_dates=["date"])
    holdout = holdout.merge(f0[KEYS + ["taiex_return"]], on=KEYS, validate="one_to_one")
    if not holdout["predicted_y_e1_0500_control"].eq(holdout["p_up"].gt(0.5).astype(int)).all():
        raise ValueError("Holdout E1 label differs from p > 0.5")
    expected_e3 = regime_predictions(holdout["p_up"].to_numpy(),
                                      holdout["taiex_return"].to_numpy(), rule)
    if not np.array_equal(holdout["predicted_y_e3_taiex_control"].to_numpy(), expected_e3):
        raise ValueError("Holdout E3 label differs from saved CV-selected rule")
    pred = holdout[KEYS + ["predicted_y_e1_0500_control",
                           "predicted_y_e3_taiex_control"]].rename(columns={
                               "predicted_y_e1_0500_control": "e1",
                               "predicted_y_e3_taiex_control": "e3"})
    periods.append(_period_rows(holdout_eval, pred, f0, history, "holdout_jan_may_2026"))

    ids, dates = load_submission_template(PROJECT_ROOT / "sample_submission.csv", target_ids)
    required = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS).to_frame(index=False)
    phase_eval = required.merge(full_history[KEYS + ["etl:adj_open", "etl:adj_close"]],
                                on=KEYS, validate="one_to_one", how="left", indicator=True)
    if not phase_eval["_merge"].eq("both").all():
        raise ValueError("Missing Phase 1 adjusted-price rows")
    phase_eval = phase_eval.drop(columns="_merge")
    e1 = _submission(PROJECT_ROOT / "submissions/zhenci_e1_phase1.csv", ids, dates)
    e3 = _submission(PROJECT_ROOT / "submissions/zhenci_e3_taiex_phase1.csv", ids, dates)
    pred = e1.rename(columns={"predicted_y": "e1"}).merge(
        e3.rename(columns={"predicted_y": "e3"}), on=KEYS, validate="one_to_one")
    periods.append(_period_rows(phase_eval, pred, f0_updated, history, "phase1_jun_sep_2026"))

    rows = pd.concat(periods, ignore_index=True)
    scores, disagreements, daily_frames = [], [], []
    for period, group in rows.groupby("period", sort=False):
        evaluate = group[KEYS + ["etl:adj_open", "etl:adj_close"]]
        metrics = {}
        for name in NAMES:
            preds = group[KEYS + [name]].rename(columns={name: "predicted_y"})
            metrics[name] = evaluate_predictions(evaluate, preds)
            scores.append({"period": period, "approach": name, **metrics[name],
                           "score_minus_candidate": metrics[name]["official_score"]
                           - group["candidate_contribution"].mean(),
                           "predicted_up_rate": float(group[name].mean())})
        for model, reference in (("e1", "candidate"), ("e3", "candidate"), ("e3", "e1")):
            part = _disagreement(group, model, reference)
            if not np.isclose(part["score_delta_on_all_rows"],
                              metrics[model]["official_score"] - metrics[reference]["official_score"],
                              rtol=0, atol=1e-14):
                raise ValueError("Disagreement contribution does not explain score delta")
            disagreements.append(part)
        daily = group.groupby("date", sort=True).agg(
            n=("stock_id", "size"),
            candidate_score=("candidate_contribution", "mean"),
            e1_score=("e1_contribution", "mean"),
            e3_score=("e3_contribution", "mean"),
            candidate_accuracy=("candidate", lambda s: float(s.eq(group.loc[s.index, "truth"]).mean())),
            e1_accuracy=("e1", lambda s: float(s.eq(group.loc[s.index, "truth"]).mean())),
            e3_accuracy=("e3", lambda s: float(s.eq(group.loc[s.index, "truth"]).mean())),
        ).reset_index()
        daily.insert(0, "period", period)
        daily_frames.append(daily)
    daily = pd.concat(daily_frames, ignore_index=True)
    scores_df = pd.DataFrame(scores)
    disagree_df = pd.DataFrame(disagreements)
    phase = rows.loc[rows["period"].eq("phase1_jun_sep_2026")]
    rule_checks = []
    for name, prediction in (
        ("prior_intraday_gt_0", phase["intraday_return_1d"].gt(0)),
        ("prior_intraday_ge_0", phase["intraday_return_1d"].ge(0)),
        ("prior_close_to_close_gt_0", phase["daily_return_1d"].gt(0)),
    ):
        if name == "prior_close_to_close_gt_0" and phase["daily_return_1d"].isna().any():
            raise ValueError("Close-to-close rule has missing Phase 1 inputs")
        preds = phase[KEYS].copy()
        preds["predicted_y"] = prediction.astype(int)
        rule_checks.append({"rule": name, **evaluate_predictions(
            phase[KEYS + ["etl:adj_open", "etl:adj_close"]], preds
        ), "predicted_up_rate": float(prediction.mean())})
    distribution = []
    for (period, approach), group in daily.melt(
        id_vars=["period", "date"], value_vars=[f"{name}_score" for name in NAMES],
        var_name="approach", value_name="daily_score"
    ).groupby(["period", "approach"], sort=False):
        x = group["daily_score"]
        distribution.append({"period": period, "approach": approach.removesuffix("_score"),
                             "days": len(x), "negative_days": int(x.lt(0).sum()),
                             "mean": float(x.mean()), "median": float(x.median()),
                             "p10": float(x.quantile(.1)), "p90": float(x.quantile(.9)),
                             "worst_day": group.loc[x.idxmin(), "date"].date().isoformat(),
                             "worst_score": float(x.min())})
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_df.to_csv(output_dir / "period_scores.csv", index=False)
    pd.DataFrame(rule_checks).to_csv(output_dir / "rule_checks.csv", index=False)
    disagree_df.to_csv(output_dir / "disagreements.csv", index=False)
    pd.DataFrame(distribution).to_csv(output_dir / "daily_distribution.csv", index=False)
    daily.to_csv(output_dir / "daily_scores.csv", index=False)
    rows[["period", *KEYS, "source_date", "intraday_return_1d", "taiex_return",
          "realized_return", "truth", *NAMES,
          *[f"{name}_contribution" for name in NAMES]]].to_csv(
              output_dir / "paired_rows.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"candidate_rule": "prior per-stock completed session adjusted close > adjusted open; ties predict 0",
                   "candidate_is_official_previous_day_definition": "unverified",
                   "e3_cv_selected_rule": rule,
                   "cv_seeds_same_probabilities": True,
                   "period_rows": {key: int(len(group)) for key, group in rows.groupby("period")},
                   "phase1_template_dates": [dates[0], dates[-1]],
                   "phase1_template_stocks": len(ids)}, file, indent=2)
        file.write("\n")
    print(scores_df.to_string(index=False))
    print(disagree_df.to_string(index=False))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)
