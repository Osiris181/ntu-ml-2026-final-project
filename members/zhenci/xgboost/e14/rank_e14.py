"""E14: rank archived E3/E12 probabilities, then optionally export a supported candidate.

Run from final_project: uv run --extra e12 python -m members.zhenci.xgboost.e14.rank_e14
The default historical mode reads saved predictions and never trains a model.
Phase 1 export is gated by the historical selection file and never uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from members.zhenci._archived_sources import archived_source_hash
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.submission import build_submission, load_submission_template, write_submission

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
ARCHIVED = ROOT / "members/zhenci/lstm/e12/outputs/validation_predictions.csv"
ARCHIVED_METRICS = ROOT / "members/zhenci/lstm/e12/outputs/fold_metrics.csv"
E12_SUMMARY = ROOT / "members/zhenci/lstm/e12/outputs/summary.json"
E1_SUMMARY = ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
E3_SUMMARY = ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
TEMPLATE = ROOT / "sample_submission.csv"
UPDATED_F0 = ROOT / "data/processed/f0_with_updates.csv"
YEARS = ("2023", "2024", "2025")
ARMS = ("e3", "e3_rank", "e12", "e12_rank")
COMPARISONS = (("e3_rank", "e3"), ("e12_rank", "e12"), ("e12_rank", "e3"))
BOOTSTRAP_DRAWS = 3000
BLOCK_DAYS = 5
BOOTSTRAP_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = {"archived_predictions": ARCHIVED, "archived_metrics": ARCHIVED_METRICS,
             "e12_summary": E12_SUMMARY, "e1_summary": E1_SUMMARY,
             "e3_summary": E3_SUMMARY, "e14_code": Path(__file__),
             "official_scorer": ROOT / "shared/evaluation.py"}
    hashes = {name: sha256(path) for name, path in paths.items()}
    hashes["e14_code"] = archived_source_hash(paths["e14_code"], hashes["e14_code"])
    return hashes


def rank_daily_25_25(rows: pd.DataFrame, probability_field: str) -> pd.DataFrame:
    """Return the input keys and 25/25 labels, ordered as the input.

    Larger probability means more likely up. Equal probabilities use ascending
    stock ID, so the result does not depend on input row order.
    """
    required = {*KEYS, probability_field}
    if required - set(rows):
        raise ValueError(f"Missing ranking inputs: {sorted(required - set(rows))}")
    keys = rows[KEYS]
    validate_keys(keys)
    probabilities = rows[probability_field].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Up probabilities must be finite and lie in [0, 1]")
    if not rows.groupby("date").size().eq(50).all():
        raise ValueError("Exact 25/25 ranking requires 50 stock rows on every date")
    ordered = rows[KEYS + [probability_field]].copy().reset_index(drop=True)
    ordered["_position"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["date", probability_field, "stock_id"],
                                  ascending=[True, False, True], kind="mergesort")
    ordered["predicted_y"] = ordered.groupby("date").cumcount().lt(25).astype(int)
    if not ordered.groupby("date")["predicted_y"].sum().eq(25).all():
        raise ValueError("The ranking rule did not assign exactly 25 up labels")
    return ordered.sort_values("_position")[KEYS + ["predicted_y"]].reset_index(drop=True)


def load_archived() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify probability direction, keys, original rules, and archived scores."""
    with E12_SUMMARY.open(encoding="utf-8") as file:
        e12_summary = json.load(file)
    with E1_SUMMARY.open(encoding="utf-8") as file:
        e1_summary = json.load(file)
    with E3_SUMMARY.open(encoding="utf-8") as file:
        e3_summary = json.load(file)
    if (e12_summary.get("input_sha256", {}).get("e1_summary") != sha256(E1_SUMMARY)
            or e12_summary.get("input_sha256", {}).get("e3_summary") != sha256(E3_SUMMARY)
            or e1_summary.get("params", {}).get("objective") != "binary:logistic"
            or e12_summary.get("training", {}).get("loss") != "BCEWithLogitsLoss"):
        raise ValueError("Archived E3 or E12 probability definitions changed")
    rule = selected_rule(e3_summary, e1_summary["config_id"])
    rows = pd.read_csv(ARCHIVED, dtype={"fold": str, "stock_id": str}, parse_dates=["date"])
    expected = {"fold", *KEYS, "etl:adj_open", "etl:adj_close", "p_up", "p_lstm",
                "e1", "e3", "e12", "taiex_return", "realized_return", "actual_up"}
    if expected - set(rows) or set(rows["fold"]) != set(YEARS):
        raise ValueError("Archived E12 comparison columns or years changed")
    if rows.duplicated(["fold", *KEYS]).any():
        raise ValueError("Duplicate archived fold/stock/date keys")
    for field in ("p_up", "p_lstm"):
        if not rows[field].between(0, 1).all():
            raise ValueError(f"Invalid {field} upward probability")
    if (not rows["e1"].eq(rows["p_up"].gt(.5).astype(int)).all()
            or not rows["e12"].eq(rows["p_lstm"].gt(.5).astype(int)).all()
            or not np.array_equal(rows["e3"].to_numpy(), regime_predictions(
                rows["p_up"].to_numpy(), rows["taiex_return"].to_numpy(), rule))):
        raise ValueError("Archived labels disagree with their upward probability rules")
    returns = (rows["etl:adj_close"] - rows["etl:adj_open"]) / rows["etl:adj_open"]
    if (not np.allclose(returns, rows["realized_return"], rtol=0, atol=1e-12)
            or not rows["actual_up"].eq(returns.gt(0).astype(int)).all()):
        raise ValueError("Archived adjusted prices or return labels changed")
    target_ids, _ = load_metadata(ROOT / "data")
    if not rows["stock_id"].isin(target_ids).all():
        raise ValueError("Archived predictions contain non-target stocks")
    archived_metrics = pd.read_csv(ARCHIVED_METRICS, dtype={"fold": str})
    for year in YEARS:
        group = rows.loc[rows["fold"].eq(year)]
        for arm in ("e3", "e12"):
            score = evaluate_predictions(group, group[KEYS + [arm]].rename(
                columns={arm: "predicted_y"}))["official_score"]
            recorded = archived_metrics.loc[archived_metrics["fold"].eq(year)
                                            & archived_metrics["arm"].eq(arm),
                                            "official_score"].iloc[0]
            if not np.isclose(score, recorded, rtol=0, atol=1e-12):
                raise ValueError(f"Archived {arm} score failed to reproduce for {year}")
    return rows, archived_metrics


def complete_date_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_ids, _ = load_metadata(ROOT / "data")
    excluded = []
    kept = []
    for (year, day), group in rows.groupby(["fold", "date"], sort=True):
        missing = sorted(set(target_ids) - set(group["stock_id"]))
        if len(group) == 50 and not missing:
            kept.append(group)
        else:
            excluded.append({"fold": year, "date": day, "available_stocks": len(group),
                             "missing_stock_ids": ",".join(missing)})
    if not kept:
        raise ValueError("No complete validation dates for 25/25 ranking")
    paired = pd.concat(kept, ignore_index=True)
    return paired, pd.DataFrame(excluded, columns=["fold", "date", "available_stocks",
                                                  "missing_stock_ids"])


def bootstrap_differences(daily: pd.DataFrame) -> pd.DataFrame:
    wide = daily.pivot(index=["fold", "date"], columns="arm", values="daily_score")
    result = []
    for candidate, control in COMPARISONS:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        annual_draws = []
        annual_points = []
        for year in YEARS:
            delta = (wide.loc[year, candidate] - wide.loc[year, control]).to_numpy()
            size = len(delta)
            draws = np.empty(BOOTSTRAP_DRAWS)
            for index in range(BOOTSTRAP_DRAWS):
                starts = rng.integers(0, size, size=(size + BLOCK_DAYS - 1) // BLOCK_DAYS)
                positions = (starts[:, None] + np.arange(BLOCK_DAYS)) % size
                draws[index] = delta[positions.ravel()[:size]].mean()
            annual_points.append(float(delta.mean()))
            annual_draws.append(draws)
            result.append({"candidate": candidate, "control": control, "period": year,
                           "score_difference": float(delta.mean()),
                           "ci95_lower": float(np.quantile(draws, .025)),
                           "ci95_upper": float(np.quantile(draws, .975))})
        equal_year = np.mean(np.stack(annual_draws), axis=0)
        result.append({"candidate": candidate, "control": control,
                       "period": "equal_year", "score_difference": float(np.mean(annual_points)),
                       "ci95_lower": float(np.quantile(equal_year, .025)),
                       "ci95_upper": float(np.quantile(equal_year, .975))})
    return pd.DataFrame(result)


def choose_candidate(metrics: pd.DataFrame, uncertainty: pd.DataFrame) -> tuple[str | None, dict]:
    """Require better mean, two supporting years, and positive paired CI."""
    means = {arm: cv_score(dict(zip(group["fold"], group["official_score"])))
             for arm, group in metrics.groupby("arm")}
    candidates = []
    checks = {}
    for candidate, own in (("e3_rank", "e3"), ("e12_rank", "e12")):
        own_by_year = metrics.loc[metrics["arm"].eq(own)].set_index("fold")["official_score"]
        ranked_by_year = metrics.loc[metrics["arm"].eq(candidate)].set_index("fold")["official_score"]
        supporting_years = [year for year in YEARS if ranked_by_year[year] > own_by_year[year]]
        ci = uncertainty.loc[uncertainty["candidate"].eq(candidate)
                             & uncertainty["control"].eq(own)
                             & uncertainty["period"].eq("equal_year"), "ci95_lower"].iloc[0]
        supported = (means[candidate] > means[own]
                     and means[candidate] > means["e3"]
                     and len(supporting_years) >= 2 and ci > 0)
        checks[candidate] = {"own_control": own, "supporting_years": supporting_years,
                             "mean_exceeds_own": bool(means[candidate] > means[own]),
                             "mean_exceeds_e3": bool(means[candidate] > means["e3"]),
                             "paired_ci_lower_positive": bool(ci > 0), "supported": bool(supported)}
        if supported:
            candidates.append(candidate)
    winner = max(candidates, key=lambda arm: means[arm]) if candidates else None
    return winner, {"equal_year_scores": means, "candidate_checks": checks}


def historical(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E14 output directory: {output_dir}")
    rows, _ = load_archived()
    paired, excluded = complete_date_rows(rows)
    for model, probability in (("e3", "p_up"), ("e12", "p_lstm")):
        for year in YEARS:
            selected = paired.loc[paired["fold"].eq(year)]
            ranked = rank_daily_25_25(selected, probability)
            paired.loc[selected.index, f"{model}_rank"] = ranked["predicted_y"].to_numpy()
        paired[f"{model}_rank"] = paired[f"{model}_rank"].astype(int)
    metric_rows, daily_rows, change_rows, quality_rows = [], [], [], []
    for year in YEARS:
        group = paired.loc[paired["fold"].eq(year)]
        for arm in ARMS:
            official = evaluate_predictions(group, group[KEYS + [arm]].rename(
                columns={arm: "predicted_y"}))
            contribution = (2 * group[arm] - 1) * group["realized_return"]
            days = pd.DataFrame({"fold": year, "date": group["date"],
                                 "arm": arm, "contribution": contribution})
            days = days.groupby(["fold", "date", "arm"], as_index=False).agg(
                daily_score=("contribution", "mean"), stock_count=("contribution", "size"))
            daily_rows.append(days)
            metric_rows.append({"fold": year, "arm": arm, **official,
                                "predicted_up_rate": float(group[arm].mean()),
                                "trading_days": len(days),
                                "positive_day_fraction": float(days["daily_score"].gt(0).mean()),
                                "negative_day_fraction": float(days["daily_score"].lt(0).mean()),
                                "median_daily_score": float(days["daily_score"].median()),
                                "p10_daily_score": float(days["daily_score"].quantile(.1)),
                                "p90_daily_score": float(days["daily_score"].quantile(.9))})
        for model, probability in (("e3", "p_up"), ("e12", "p_lstm")):
            ranked_name = f"{model}_rank"
            changed = group[ranked_name].ne(group[model])
            up_switch = group[model].eq(0) & group[ranked_name].eq(1)
            down_switch = group[model].eq(1) & group[ranked_name].eq(0)
            delta = 2 * (group[ranked_name] - group[model]) * group["realized_return"]
            change_rows.append({"fold": year, "model": model, "changed_predictions": int(changed.sum()),
                                "zero_to_one": int(up_switch.sum()),
                                "one_to_zero": int(down_switch.sum()),
                                "score_difference_from_changes": float(delta.mean()),
                                "score_difference_on_changed_rows": float(delta[changed].mean())})
            for day, date_rows in group.groupby("date"):
                top = date_rows.loc[date_rows[ranked_name].eq(1), "realized_return"]
                bottom = date_rows.loc[date_rows[ranked_name].eq(0), "realized_return"]
                spread = float(top.mean() - bottom.mean())
                quality_rows.append({"fold": year, "date": day, "model": model,
                                     "spearman_probability_return": float(date_rows[
                                         [probability, "realized_return"]].corr(
                                             method="spearman").iloc[0, 1]),
                                     "top25_mean_return": float(top.mean()),
                                     "bottom25_mean_return": float(bottom.mean()),
                                     "top_minus_bottom_return": spread,
                                     "positive_spread": int(spread > 0)})
    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    changes = pd.DataFrame(change_rows)
    quality = pd.DataFrame(quality_rows)
    uncertainty = bootstrap_differences(daily)
    winner, checks = choose_candidate(metrics, uncertainty)
    selection = {"selected_candidate": winner, "decision_rule":
                 "Ranked mean > own original and E3; gain in at least two years; "
                 "paired five-day-block 95% lower bound > 0 versus own original",
                 "complete_dates_only": True, "n_complete_dates": int(paired.groupby(
                     ["fold", "date"]).ngroups), "n_excluded_dates": len(excluded),
                 "input_sha256": source_hashes(), **checks}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("historical_predictions", paired), ("historical_metrics", metrics),
                        ("daily_scores", daily), ("ranking_quality", quality),
                        ("changed_predictions", changes), ("paired_uncertainty", uncertainty),
                        ("excluded_incomplete_dates", excluded)):
        table.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2, allow_nan=False)
        file.write("\n")
    print(metrics[["fold", "arm", "official_score", "accuracy",
                   "positive_day_fraction"]].to_string(index=False), flush=True)
    print(f"Selected Phase 1 ranking candidate: {winner or 'none'}", flush=True)
    return output_dir


def phase1_probabilities(candidate: str, f0_path: Path, template_path: Path,
                         data_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load only an existing frozen final model and its fitted preprocessor."""
    from members.zhenci.xgboost.e1.predict_e1 import require_template_features
    from shared.features import load_f0_dataset
    from shared.preprocessing import F0_FEATURES

    target_ids, _ = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    if len(dates) != 84 or len(ids) != 50 or min(dates) <= "2025-12-31":
        raise ValueError("Expected the official 84-date Phase 1 template after 2025")
    f0 = load_f0_dataset(f0_path)
    matched = require_template_features(f0, ids, dates)
    keys = matched[KEYS].reset_index(drop=True)
    if candidate == "e3_rank":
        import xgboost as xgb
        from members.zhenci.xgboost.e1.save_e1 import DEFAULT_OUTPUT_DIR as artifact_dir
        from shared.preprocessing import BaselinePreprocessor

        with (artifact_dir / "manifest.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
        with E3_SUMMARY.open(encoding="utf-8") as file:
            e3_summary = json.load(file)
        selected_rule(e3_summary, manifest["config_id"])
        if (manifest.get("artifact_version") != 1 or manifest.get("model") != "XGBoost"
                or manifest.get("train_end") != "2025-12-31"
                or manifest.get("params", {}).get("objective") != "binary:logistic"
                or manifest.get("feature_order") != list(F0_FEATURES)):
            raise ValueError("Unexpected frozen E3 source model")
        for name, field in (("model.json", "model_sha256"),
                            ("preprocessor.json", "preprocessor_sha256"),
                            ("feature_definitions.py", "feature_definitions_sha256"),
                            ("preprocessing_definitions.py", "preprocessing_definitions_sha256")):
            if sha256(artifact_dir / name) != manifest.get(field):
                raise ValueError(f"Changed E3 source artifact file: {name}")
        if (sha256(ROOT / "shared/features.py") != manifest["feature_definitions_sha256"]
                or sha256(ROOT / "shared/preprocessing.py")
                != manifest["preprocessing_definitions_sha256"]):
            raise ValueError("Current F0 or preprocessing differs from frozen E3 source")
        with (artifact_dir / "preprocessor.json").open(encoding="utf-8") as file:
            preprocessor = BaselinePreprocessor.from_state(json.load(file))
        features = preprocessor.transform(matched[list(F0_FEATURES)])
        if list(features.columns) != manifest["transformed_feature_order"]:
            raise ValueError("E3 frozen feature order changed")
        model = xgb.Booster(model_file=str(artifact_dir / "model.json"))
        probability = model.predict(xgb.DMatrix(features))
    elif candidate == "e12_rank":
        from members.zhenci.lstm.e12.predict_e12 import load_artifact
        from members.zhenci.lstm.e12.save_e12 import DEFAULT_ARTIFACT
        from members.zhenci.lstm.e12.sequences import sequence_batch
        from members.zhenci.lstm.e12.train_e12 import _probabilities

        manifest, preprocessor, model = load_artifact(DEFAULT_ARTIFACT)
        batch, context_names = sequence_batch(f0, keys, preprocessor)
        if (not batch["keys"].equals(keys)
                or context_names != manifest["context_names"]):
            raise ValueError("E12 frozen sequence keys or context changed")
        probability = _probabilities(model, batch)
    else:
        raise ValueError("Unsupported ranking candidate")
    result = keys.copy()
    result["p_up"] = probability
    return result, ids, dates


def phase1(output_dir: Path, f0_path: Path, template_path: Path,
           data_dir: Path, output_path: Path | None) -> Path:
    with (output_dir / "selection.json").open(encoding="utf-8") as file:
        selection = json.load(file)
    if selection.get("input_sha256") != source_hashes():
        raise ValueError("E14 historical selection inputs or code changed")
    candidate = selection.get("selected_candidate")
    if candidate not in {"e3_rank", "e12_rank"}:
        raise ValueError("Historical E14 selected no ranking candidate; no Phase 1 export")
    probabilities, ids, dates = phase1_probabilities(candidate, f0_path,
                                                     template_path, data_dir)
    predictions = rank_daily_25_25(probabilities, "p_up")
    if (len(predictions) != 4200 or predictions.groupby("date")["predicted_y"].sum().ne(25).any()
            or predictions.groupby("date").size().ne(50).any()):
        raise ValueError("Phase 1 ranking does not contain 84 complete 25/25 dates")
    submission = build_submission(predictions, template_path, ids)
    if (list(submission.columns) != ["id", *dates]
            or submission["id"].tolist() != ids
            or not submission[dates].sum(axis=0).eq(25).all()):
        raise ValueError("Phase 1 submission layout or daily counts changed")
    destination = output_path or ROOT / "submissions" / f"zhenci_e14_{candidate}_phase1.csv"
    saved = write_submission(predictions, template_path, destination, ids)
    restored = pd.read_csv(saved, dtype={"id": str})
    if not restored.equals(submission):
        raise ValueError("Saved Phase 1 submission differs from validated in-memory CSV")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("historical", "phase1"), default="historical")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.mode == "historical":
        print(historical(args.output_dir))
    else:
        print(phase1(args.output_dir, args.f0_path, args.template,
                     args.data_dir, args.submission))
