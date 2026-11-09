"""E8: chronological, confidence-limited previous-direction override.

Run from the repository root:
    uv run python -m members.zhenci.e8.evaluate_e8

Reads saved out-of-sample probabilities. No model fitting or threshold search
beyond the four declared override distances. Refuses to overwrite results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset, stock_history_candidates
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import E1_SEEDS, FOLDS, split_market
from shared.updated_features import load_updated_market

OUTPUT = PROJECT_ROOT / "members/zhenci/e8/outputs"
DISTANCES = (0.0, 0.01, 0.025, 0.05)
YEARS = tuple(f.name for f in FOLDS)
BLOCK_LENGTH = 5
BOOTSTRAP_DRAWS = 5000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold(taiex: pd.Series, rule: dict) -> np.ndarray:
    x = taiex.to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError("Infinite TAIEX return")
    return np.where(np.isnan(x), rule["missing"],
                    np.where(x < 0, rule["negative"], rule["nonnegative"]))


def _e8(base: pd.DataFrame, distance: float) -> tuple[np.ndarray, np.ndarray]:
    if distance not in DISTANCES:
        raise ValueError("Undeclared E8 distance")
    override = (base["previous_direction"].eq(1).to_numpy()
                & base["e3"].eq(0).to_numpy()
                & ((base["threshold"] - base["p_up"]).to_numpy() <= distance)
                & (distance > 0))
    result = np.where(override, 1, base["e3"].to_numpy(dtype=int))
    return result.astype(int), override


def _official(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    pred = frame[KEYS].copy()
    pred["predicted_y"] = predictions.astype(int)
    return evaluate_predictions(frame, pred)


def _base(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    validate_keys(frame)
    required = {"etl:adj_open", "etl:adj_close", "intraday_return_1d",
                "taiex_return", "p_up"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if (frame["etl:adj_open"].le(0).any()
            or frame[["etl:adj_open", "etl:adj_close", "intraday_return_1d", "p_up"]]
            .isna().any().any()):
        raise ValueError("Invalid prices, probabilities, or previous return")
    if not frame["p_up"].between(0, 1).all():
        raise ValueError("Invalid probabilities")
    out = frame.copy()
    out["threshold"] = _threshold(out["taiex_return"], rule)
    out["previous_direction"] = out["intraday_return_1d"].gt(0).astype(int)
    out["e1"] = out["p_up"].gt(0.5).astype(int)
    out["e3"] = out["p_up"].gt(out["threshold"]).astype(int)
    out["realized_return"] = ((out["etl:adj_close"] - out["etl:adj_open"])
                               / out["etl:adj_open"])
    out["truth"] = out["realized_return"].gt(0).astype(int)
    out["taiex_condition"] = np.select(
        [out["taiex_return"].isna(), out["taiex_return"].lt(0)],
        ["missing", "negative"], default="nonnegative")
    out["distance_below_threshold"] = out["threshold"] - out["p_up"]
    return out


def _historical(market: pd.DataFrame, f0: pd.DataFrame, target_ids: list[str],
                rule: dict) -> dict[str, pd.DataFrame]:
    path = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"
    source = pd.read_csv(path, dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    if (set(source["fold"]) != set(YEARS) or set(source["seed"]) != set(E1_SEEDS)
            or source["config_id"].nunique() != 1
            or source.duplicated(["fold", "seed", *KEYS]).any()):
        raise ValueError("Saved OOS prediction contract failed")
    if source.groupby(["fold", *KEYS])["p_up"].nunique().ne(1).any():
        raise ValueError("Saved seed probabilities differ; one-seed shortcut invalid")
    if not source["predicted_y"].eq(source["p_up"].gt(.5).astype(int)).all():
        raise ValueError("Saved E1 labels do not match probabilities")
    source = source.loc[source["seed"].eq(42)].copy()
    frames = {}
    for fold in FOLDS:
        train, valid = split_market(market, target_ids, fold)
        if train["date"].max() >= valid["date"].min():
            raise ValueError("Fold training/evaluation dates overlap")
        pred = source.loc[source["fold"].eq(fold.name), KEYS + ["p_up"]]
        if len(pred) != len(valid):
            raise ValueError(f"Missing OOS rows for {fold.name}")
        rows = valid[KEYS + ["etl:adj_open", "etl:adj_close"]].merge(
            pred, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not rows["_merge"].eq("both").all():
            raise ValueError(f"OOS keys do not match {fold.name}")
        rows = rows.drop(columns="_merge").merge(
            f0[KEYS + ["intraday_return_1d", "taiex_return"]],
            on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not rows["_merge"].eq("both").all():
            raise ValueError(f"F0 keys do not match {fold.name}")
        rows = _base(rows.drop(columns="_merge"), rule)
        rows.insert(0, "period", fold.name)
        frames[fold.name] = rows.sort_values(["date", "stock_id"]).reset_index(drop=True)
    return frames


def _disagreement_bins(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for period, frame in frames.items():
        changed = frame.loc[frame["previous_direction"].eq(1) & frame["e3"].eq(0)].copy()
        if changed.empty or changed["distance_below_threshold"].lt(0).any():
            raise ValueError("Invalid disagreement population")
        changed["distance_band"] = pd.cut(
            changed["distance_below_threshold"],
            bins=[-1e-12, .01, .025, .05, np.inf],
            labels=["[0,.01]", "(.01,.025]", "(.025,.05]", ">.05"],
            include_lowest=True)
        if changed["distance_band"].isna().any():
            raise ValueError("Missing disagreement band")
        changed["if_override_delta"] = 2 * changed["realized_return"]
        for (band, regime), group in changed.groupby(
            ["distance_band", "taiex_condition"], observed=True, sort=True):
            parts.append({"period": period, "distance_band": str(band),
                          "taiex_condition": regime, "n_disagreements": len(group),
                          "fraction_of_fold": len(group) / len(frame),
                          "mean_realized_return": float(group["realized_return"].mean()),
                          "median_realized_return": float(group["realized_return"].median()),
                          "positive_return_rate": float(group["realized_return"].gt(0).mean()),
                          "e3_contribution_sum": float((-group["realized_return"]).sum()),
                          "override_contribution_sum": float(group["realized_return"].sum()),
                          "if_override_delta_sum": float(group["if_override_delta"].sum()),
                          "if_override_fold_score_delta": float(group["if_override_delta"].sum()
                                                           / len(frame))})
    return pd.DataFrame(parts)


def _grid(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for year, frame in frames.items():
        for distance in DISTANCES:
            prediction, mask = _e8(frame, distance)
            metrics = _official(frame, prediction)
            delta_sum = float((2 * frame.loc[mask, "realized_return"]).sum())
            original_score = _official(frame, frame["e3"].to_numpy())["official_score"]
            if not np.isclose(metrics["official_score"] - original_score,
                              delta_sum / len(frame), rtol=0, atol=1e-14):
                raise ValueError("Override contribution does not explain score change")
            records.append({"period": year, "distance": distance, **metrics,
                            "up_fraction": float(prediction.mean()),
                            "n_overrides": int(mask.sum()),
                            "override_contribution_sum": delta_sum,
                            "score_delta_vs_e3": float(metrics["official_score"] - original_score)})
    return pd.DataFrame(records)


def _select(grid: pd.DataFrame, earlier: tuple[str, ...]) -> float:
    if not earlier:
        return 0.0
    if any(year not in YEARS for year in earlier):
        raise ValueError("Nonhistorical selection year")
    candidates = []
    for distance in DISTANCES:
        scores = grid.loc[grid["distance"].eq(distance)
                          & grid["period"].isin(earlier)].set_index("period")
        if set(scores.index) != set(earlier):
            raise ValueError("Missing earlier OOS score")
        candidates.append((float(scores.loc[list(earlier), "official_score"].mean()), distance))
    # Exact ties retain the smaller override distance, beginning with zero.
    best = max(score for score, _ in candidates)
    return min(distance for score, distance in candidates if np.isclose(score, best,
                                                                        rtol=0, atol=1e-15))


def _daily_and_metrics(frames: dict[str, pd.DataFrame], choices: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_parts, metrics = [], []
    for period, frame in frames.items():
        e8, mask = _e8(frame, choices[period])
        work = frame[KEYS + ["realized_return", "truth", "e1", "e3", "previous_direction"]].copy()
        work["e8"] = e8
        work["override"] = mask
        for name in ("e1", "e3", "previous_direction", "e8"):
            pred = work[name].to_numpy(dtype=int)
            score = _official(frame, pred)
            metrics.append({"period": period, "approach": name, **score,
                            "up_fraction": float(pred.mean()),
                            "n_overrides": int(mask.sum()) if name == "e8" else 0,
                            "override_contribution_sum": float((2 * work.loc[mask, "realized_return"]).sum())
                            if name == "e8" else 0.0})
            work[f"{name}_contribution"] = (2 * pred - 1) * work["realized_return"]
        work["e8_minus_e3_contribution"] = (work["e8_contribution"]
                                              - work["e3_contribution"])
        daily = work.groupby("date", sort=True).agg(
            n=("stock_id", "size"),
            e1_score=("e1_contribution", "mean"),
            e3_score=("e3_contribution", "mean"),
            previous_direction_score=("previous_direction_contribution", "mean"),
            e8_score=("e8_contribution", "mean"),
            e8_minus_e3_sum=("e8_minus_e3_contribution", "sum"),
            n_overrides=("override", "sum"),
        ).reset_index()
        daily.insert(0, "period", period)
        daily_parts.append(daily)
    return pd.concat(daily_parts, ignore_index=True), pd.DataFrame(metrics)


def _paired_block_bootstrap(daily: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    draws_by_year = {}
    records = []
    for period in (*YEARS, "reserved_oct_2026"):
        d = daily.loc[daily["period"].eq(period)].sort_values("date")
        n = len(d)
        starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, int(np.ceil(n / BLOCK_LENGTH))))
        indices = ((starts[:, :, None] + np.arange(BLOCK_LENGTH)) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]
        numer = d["e8_minus_e3_sum"].to_numpy()[indices].sum(axis=1)
        denom = d["n"].to_numpy()[indices].sum(axis=1)
        draws = numer / denom
        draws_by_year[period] = draws
        observed = float(d["e8_minus_e3_sum"].sum() / d["n"].sum())
        records.append({"period": period, "observed_delta": observed,
                        "ci_2p5": float(np.quantile(draws, .025)),
                        "ci_97p5": float(np.quantile(draws, .975)),
                        "bootstrap_p_delta_positive": float((draws > 0).mean())})
    pooled = np.mean(np.stack([draws_by_year[y] for y in YEARS]), axis=0)
    records.append({"period": "equal_year_cv_mean",
                    "observed_delta": float(np.mean([r["observed_delta"] for r in records if r["period"] in YEARS])),
                    "ci_2p5": float(np.quantile(pooled, .025)),
                    "ci_97p5": float(np.quantile(pooled, .975)),
                    "bootstrap_p_delta_positive": float((pooled > 0).mean())})
    return pd.DataFrame(records)


def _reserved(rule: dict, final_distance: float, target_ids: list[str],
              all_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One later-period evaluation, reached only after final distance freezes."""
    artifact = PROJECT_ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
    with (artifact / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("train_end") != "2025-12-31"
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("threshold") != .5):
        raise ValueError("Frozen artifact contract changed")
    for name, field in (("model.json", "model_sha256"),
                        ("preprocessor.json", "preprocessor_sha256"),
                        ("feature_definitions.py", "feature_definitions_sha256"),
                        ("preprocessing_definitions.py", "preprocessing_definitions_sha256")):
        if _sha(artifact / name) != manifest.get(field):
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    if (_sha(PROJECT_ROOT / "shared/features.py") != manifest["feature_definitions_sha256"]
            or _sha(PROJECT_ROOT / "shared/preprocessing.py")
            != manifest["preprocessing_definitions_sha256"]):
        raise ValueError("Shared F0/preprocessing changed from frozen artifact")
    f0 = load_f0_dataset(PROJECT_ROOT / "data/processed/f0_with_updates.csv")
    update = load_updated_market(PROJECT_ROOT / "updated_data_7_Nov/data", all_ids)
    evaluate = update.loc[update["stock_id"].isin(target_ids)
                          & update["date"].between("2026-10-01", "2026-10-22")].copy()
    if (len(evaluate) != 750 or evaluate["date"].nunique() != 15
            or not evaluate.groupby("date").size().eq(50).all()):
        raise ValueError("Reserved October window is incomplete")
    if (evaluate["etl:adj_open"].le(0).any()
            or evaluate[["etl:adj_open", "etl:adj_close"]].isna().any().any()):
        raise ValueError("Reserved October prices invalid")
    rows = evaluate[KEYS + ["etl:adj_open", "etl:adj_close"]].merge(
        f0, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not rows["_merge"].eq("both").all():
        raise ValueError("Reserved F0 rows missing")
    rows = rows.drop(columns="_merge").sort_values(["date", "stock_id"]).reset_index(drop=True)
    with (artifact / "preprocessor.json").open(encoding="utf-8") as file:
        processor = BaselinePreprocessor.from_state(json.load(file))
    transformed = processor.transform(rows.loc[:, list(F0_FEATURES)])
    if transformed.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("Frozen transformed order differs")
    booster = xgb.Booster(model_file=str(artifact / "model.json"))
    p_up = booster.predict(xgb.DMatrix(transformed))
    rows["p_up"] = p_up
    rows = _base(rows, rule)
    # Recheck the previous-session signal from the raw original + updated data.
    original = load_market(PROJECT_ROOT / "data")
    history = stock_history_candidates(pd.concat([original.drop(columns="y"), update],
                                                  ignore_index=True))
    prior = rows[KEYS + ["intraday_return_1d"]].merge(
        history[KEYS + ["source_date", "intraday_return_1d"]],
        on=KEYS, validate="one_to_one", suffixes=("", "_raw"))
    if (not prior["source_date"].lt(prior["date"]).all()
            or not np.allclose(prior["intraday_return_1d"], prior["intraday_return_1d_raw"],
                               rtol=1e-10, atol=1e-12, equal_nan=True)):
        raise ValueError("Reserved previous-session alignment invalid")
    rows.insert(0, "period", "reserved_oct_2026")
    daily, metrics = _daily_and_metrics({"reserved_oct_2026": rows},
                                         {"reserved_oct_2026": final_distance})
    return daily, metrics


def run(output_dir: Path = OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Results already exist: {output_dir}")
    data_dir = PROJECT_ROOT / "data"
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(data_dir / "processed/f0.csv")
    summary_path = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    rule = selected_rule(summary, "xgboost_initial_v1")
    frames = _historical(market, f0, target_ids, rule)
    bins = _disagreement_bins(frames)
    grid = _grid(frames)
    choices = {year: _select(grid, YEARS[:i]) for i, year in enumerate(YEARS)}
    daily, metrics = _daily_and_metrics(frames, choices)
    # Final configuration is selected using all three completed OOS years.
    final_distance = _select(grid, YEARS)
    # The October files are first read/scored after final_distance is frozen.
    reserved_daily, reserved_metrics = _reserved(rule, final_distance, target_ids, all_ids)
    uncertainty = _paired_block_bootstrap(pd.concat([daily, reserved_daily], ignore_index=True))
    distribution = []
    for (period, approach), group in pd.concat([daily, reserved_daily]).melt(
        id_vars=["period", "date"],
        value_vars=["e1_score", "e3_score", "previous_direction_score", "e8_score"],
        var_name="approach", value_name="score").groupby(["period", "approach"], sort=False):
        scores = group["score"]
        distribution.append({"period": period, "approach": approach.removesuffix("_score"),
                             "days": len(scores), "negative_days": int(scores.lt(0).sum()),
                             "daily_median": float(scores.median()),
                             "daily_p10": float(scores.quantile(.1)),
                             "daily_p90": float(scores.quantile(.9))})
    output_dir.mkdir(parents=True, exist_ok=False)
    bins.to_csv(output_dir / "historical_disagreement_bins.csv", index=False)
    grid.to_csv(output_dir / "candidate_grid.csv", index=False)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    daily.to_csv(output_dir / "daily_scores.csv", index=False)
    uncertainty.to_csv(output_dir / "paired_uncertainty.csv", index=False)
    pd.DataFrame(distribution).to_csv(output_dir / "daily_distribution.csv", index=False)
    reserved_metrics.to_csv(output_dir / "reserved_metrics.csv", index=False)
    reserved_daily.to_csv(output_dir / "reserved_daily_scores.csv", index=False)
    with (output_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump({"candidate_distances": DISTANCES, "choices_using_only_earlier_oos_years": choices,
                   "final_distance_selected_from_2023_to_2025": final_distance,
                   "e3_fixed_rule": rule,
                   "reserved_period": "2026-10-01 to 2026-10-22; 15 sessions; excludes 2026-10-23 sentinel",
                   "bootstrap": {"type": "paired circular moving-block by trading day",
                                 "block_sessions": BLOCK_LENGTH, "draws": BOOTSTRAP_DRAWS,
                                 "seed": 42},
                   "inputs_sha256": {
                       "saved_cv_predictions": _sha(PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"),
                       "e3_summary": _sha(summary_path),
                       "historical_f0": _sha(data_dir / "processed/f0.csv"),
                       "updated_f0": _sha(data_dir / "processed/f0_with_updates.csv"),
                       "frozen_artifact_manifest": _sha(PROJECT_ROOT / "results/zhenci_e1_full_pre_holdout_artifact/manifest.json")}},
                  file, indent=2)
        file.write("\n")
    print(json.dumps({"choices": choices, "final_distance": final_distance}, indent=2))
    print(metrics.to_string(index=False))
    print(reserved_metrics.to_string(index=False))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)
