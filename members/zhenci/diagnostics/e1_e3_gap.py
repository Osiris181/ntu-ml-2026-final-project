"""Reproduce and audit E1/E3 Phase 1 predictions without retraining.

From final_project: uv run python -m members.zhenci.diagnostics.e1_e3_gap
Writes only new diagnostic files beside this script; never changes experiments.
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
import xgboost as xgb

from members.zhenci.e1.predict_e1 import require_template_features
from members.zhenci.e1.save_e1 import _sha256
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_global, load_market, load_metadata
from shared.evaluation import evaluate_predictions
from shared.features import (
    GLOBAL_MAX_AGE_DAYS, GLOBAL_SERIES, build_f0_dataset, load_f0_dataset,
    stock_history_candidates,
)
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT, split_market
from shared.submission import load_submission_template
from shared.updated_features import combine_feature_history, load_updated_market

OUTPUT_DIR = Path(__file__).resolve().parent / "e1_e3_gap_outputs"


def _submitted(path: Path, ids: list[str], dates: list[str]) -> pd.DataFrame:
    wide = pd.read_csv(path, dtype={"id": str})
    if wide.columns.tolist() != ["id", *dates] or wide["id"].tolist() != ids:
        raise ValueError(f"Submission does not match the official template: {path}")
    values = wide.loc[:, dates]
    if values.isna().any().any() or not values.isin([0, 1]).all().all():
        raise ValueError(f"Submission is not complete binary 0/1: {path}")
    long = wide.melt(id_vars="id", var_name="date", value_name="predicted_y")
    long = long.rename(columns={"id": "stock_id"})
    long["date"] = pd.to_datetime(long["date"])
    return long


def _same_numeric(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    return all(np.allclose(left[name].to_numpy(dtype=float), right[name].to_numpy(dtype=float),
                           rtol=1e-10, atol=1e-12, equal_nan=True) for name in columns)


def run(output_dir: Path = OUTPUT_DIR) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output exists; choose a new path: {output_dir}")
    base = PROJECT_ROOT / "data"
    update = PROJECT_ROOT / "updated_data_7_Nov/data"
    f0_path = PROJECT_ROOT / "data/processed/f0.csv"
    updated_f0_path = PROJECT_ROOT / "data/processed/f0_with_updates.csv"
    artifact = PROJECT_ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
    e3_summary_path = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
    template = PROJECT_ROOT / "sample_submission.csv"
    e1_file = PROJECT_ROOT / "submissions/zhenci_e1_phase1.csv"
    e3_file = PROJECT_ROOT / "submissions/zhenci_e3_taiex_phase1.csv"
    target_ids, all_ids = load_metadata(base)
    ids, dates = load_submission_template(template, target_ids)
    keys = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS).to_frame(index=False)
    if len(keys) != 4200:
        raise ValueError("Expected exactly 50 stocks x 84 Phase 1 dates")

    original_market = load_market(base)
    updated_market = load_updated_market(update, all_ids)
    combined, global_data = combine_feature_history(
        original_market, updated_market, load_global(base), load_global(update)
    )
    realized = keys.merge(combined.drop(columns="y", errors="ignore"), on=KEYS,
                          how="left", validate="one_to_one", indicator=True)
    if not realized["_merge"].eq("both").all():
        raise ValueError("Realized market rows do not cover all template keys")
    realized = realized.drop(columns="_merge")
    if (realized[["etl:adj_open", "etl:adj_close"]].isna().any().any()
            or (realized["etl:adj_open"] <= 0).any()):
        raise ValueError("Realized returns are incomplete or invalid")

    with (artifact / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("training_split") != HOLDOUT.name
            or manifest.get("train_end") != HOLDOUT.train_end
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("threshold") != 0.5
            or _sha256(f0_path) != manifest.get("f0_csv_sha256")):
        raise ValueError("Frozen E1 artifact does not match the 2020-2025 F0 training contract")
    for filename, field in (
        ("model.json", "model_sha256"),
        ("preprocessor.json", "preprocessor_sha256"),
        ("feature_definitions.py", "feature_definitions_sha256"),
        ("preprocessing_definitions.py", "preprocessing_definitions_sha256"),
    ):
        if _sha256(artifact / filename) != manifest.get(field):
            raise ValueError(f"Frozen artifact changed: {filename}")
    if (_sha256(PROJECT_ROOT / "shared/features.py") != manifest["feature_definitions_sha256"]
            or _sha256(PROJECT_ROOT / "shared/preprocessing.py") != manifest["preprocessing_definitions_sha256"]):
        raise ValueError("Current shared F0 or preprocessing differs from the frozen artifact")

    original_f0 = load_f0_dataset(f0_path)
    updated_f0 = load_f0_dataset(updated_f0_path)
    rows = require_template_features(updated_f0, ids, dates)
    fresh = require_template_features(build_f0_dataset(combined, global_data), ids, dates)
    numeric = [field for field in F0_FEATURES if field not in ("stock_id", "day_of_week")]
    if not _same_numeric(rows, fresh, numeric) or not rows["day_of_week"].equals(fresh["day_of_week"]):
        raise ValueError("Saved template F0 does not match a fresh build from supplied sources")
    history = stock_history_candidates(combined)
    source_dates = keys.merge(history[KEYS + ["source_date", "institutional_source_date"]],
                             on=KEYS, validate="one_to_one")
    if (source_dates["source_date"].isna().any()
            or source_dates["institutional_source_date"].isna().any()
            or not (source_dates["source_date"] < source_dates["date"]).all()
            or not (source_dates["institutional_source_date"] < source_dates["source_date"]).all()):
        raise ValueError("Stock lag dates violate t-1/t-2 ordering")
    for feature, series in GLOBAL_SERIES.items():
        observed = global_data.loc[:, ["date", series]].dropna().sort_values("date").copy()
        observed["independent_value"] = (
            observed[series] / 100 if feature == "vix_normalized"
            else observed[series] / observed[series].shift(1) - 1
        )
        eligible = pd.merge_asof(
            pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date"),
            observed.loc[:, ["date", "independent_value"]].rename(columns={"date": "source_date"}),
            left_on="date", right_on="source_date", direction="backward",
            allow_exact_matches=False, tolerance=pd.Timedelta(days=GLOBAL_MAX_AGE_DAYS),
        )
        actual = rows.groupby("date", sort=True)[feature].first().reset_index()
        eligible = eligible.merge(actual, on="date", validate="one_to_one")
        if (not np.allclose(eligible["independent_value"], eligible[feature],
                            equal_nan=True, rtol=1e-10, atol=1e-12)
                or (eligible["source_date"] >= eligible["date"]).any()
                or ((eligible["date"] - eligible["source_date"]).dt.days > GLOBAL_MAX_AGE_DAYS).any()):
            raise ValueError(f"Global feature {feature} is not independently strict-prior aligned")

    train, _ = split_market(original_market, target_ids, HOLDOUT)
    if (len(train) != manifest["training_rows"] or train["date"].max() > pd.Timestamp(HOLDOUT.train_end)
            or set(train["stock_id"]) != set(all_ids)
            or not train["y"].eq((train["etl:adj_close"] > train["etl:adj_open"]).astype(int)).all()):
        raise ValueError("E1 training rows or labels differ from frozen artifact contract")
    train_f0 = train.loc[:, KEYS].merge(original_f0, on=KEYS, validate="one_to_one")
    with (artifact / "preprocessor.json").open("r", encoding="utf-8") as file:
        frozen_state = json.load(file)
    frozen_preprocessor = BaselinePreprocessor.from_state(frozen_state)
    audit_fit = BaselinePreprocessor().fit(train_f0.loc[:, list(F0_FEATURES)])
    if (frozen_state["categories"] != {name: list(v) for name, v in audit_fit.categories_.items()}
            or frozen_state["feature_names"] != list(audit_fit.feature_names_)
            or any(not np.allclose(frozen_state[name], getattr(audit_fit, attr),
                                   rtol=1e-12, atol=1e-12)
                   for name, attr in (("medians", "medians_"), ("means", "means_"), ("scales", "scales_")))):
        raise ValueError("Frozen preprocessing differs from fitting on 2020-2025 training rows")
    transformed = frozen_preprocessor.transform(rows.loc[:, list(F0_FEATURES)])
    if transformed.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("Frozen transformed feature order differs from model artifact")
    booster = xgb.Booster(model_file=str(artifact / "model.json"))
    probabilities = booster.predict(xgb.DMatrix(transformed))
    with e3_summary_path.open("r", encoding="utf-8") as file:
        rule = selected_rule(json.load(file), manifest["config_id"])
    if (rule["negative"], rule["nonnegative"], rule["missing"]) != (0.525, 0.475, 0.5):
        raise ValueError("Unexpected E3 decision rule")
    expected_e1 = (probabilities > 0.5).astype(int)
    expected_e3 = regime_predictions(probabilities, rows["taiex_return"].to_numpy(), rule)
    e1 = _submitted(e1_file, ids, dates).rename(columns={"predicted_y": "e1_pred"})
    e3 = _submitted(e3_file, ids, dates).rename(columns={"predicted_y": "e3_pred"})
    paired = rows.loc[:, KEYS + ["taiex_return"]].copy()
    paired["p_up"] = probabilities
    paired["expected_e1"] = expected_e1
    paired["expected_e3"] = expected_e3
    paired = paired.merge(e1, on=KEYS, validate="one_to_one").merge(e3, on=KEYS, validate="one_to_one")
    if (not paired["expected_e1"].eq(paired["e1_pred"]).all()
            or not paired["expected_e3"].eq(paired["e3_pred"]).all()):
        raise ValueError("Submitted predictions do not reproduce from frozen model and F0")
    paired = paired.merge(realized[KEYS + ["etl:adj_open", "etl:adj_close"]],
                          on=KEYS, validate="one_to_one")
    paired["realized_return"] = paired["etl:adj_close"] / paired["etl:adj_open"] - 1
    paired["e1_contribution"] = (2 * paired["e1_pred"] - 1) * paired["realized_return"]
    paired["e3_contribution"] = (2 * paired["e3_pred"] - 1) * paired["realized_return"]
    paired["delta_contribution"] = paired["e3_contribution"] - paired["e1_contribution"]
    paired["disagreement"] = paired["e1_pred"].ne(paired["e3_pred"])
    paired["taiex_regime"] = np.select(
        [paired["taiex_return"].isna(), paired["taiex_return"].lt(0)],
        ["missing", "negative"], default="nonnegative",
    )
    official = {}
    for name, labels in (("E1", paired["e1_pred"]), ("E3", paired["e3_pred"])):
        predictions = paired[KEYS].copy()
        predictions["predicted_y"] = labels
        official[name] = evaluate_predictions(realized, predictions)
    if (not np.isclose(official["E1"]["official_score"], paired["e1_contribution"].mean())
            or not np.isclose(official["E3"]["official_score"], paired["e3_contribution"].mean())):
        raise ValueError("Daily decomposition does not agree with official scorer")
    daily = paired.groupby("date", sort=True).agg(
        n=("stock_id", "size"), taiex_regime=("taiex_regime", "first"),
        mean_realized_return=("realized_return", "mean"),
        e1_score=("e1_contribution", "mean"), e3_score=("e3_contribution", "mean"),
        e3_minus_e1=("delta_contribution", "mean"),
        disagreements=("disagreement", "sum"),
    ).reset_index()
    if not daily["n"].eq(50).all():
        raise ValueError("Every template day should have 50 target stocks")
    negative = -daily.loc[daily["e3_score"] < 0, "e3_score"]
    summary = {
        "template_dates": [dates[0], dates[-1]],
        "template_stocks": len(ids), "template_days": len(dates),
        "realized_rows": len(realized),
        "original_market_last_date": original_market["date"].max().date().isoformat(),
        "updated_market_first_date": updated_market["date"].min().date().isoformat(),
        "f0_rebuilt_template_numeric_fields_match": True,
        "stock_lag_source_violations": 0,
        "global_strict_prior_or_staleness_violations": 0,
        "preprocessing_refit_on_training_rows_matches_frozen_state": True,
        "submitted_predictions_mismatching_frozen_model": 0,
        "artifact_manifest_sha256": _sha256(artifact / "manifest.json"),
        "updated_f0_sha256": _sha256(updated_f0_path),
        "e1_submission_sha256": _sha256(e1_file),
        "e3_submission_sha256": _sha256(e3_file),
        "e1": official["E1"], "e3": official["E3"],
        "e3_minus_e1_official_score": official["E3"]["official_score"] - official["E1"]["official_score"],
        "disagreements": int(paired["disagreement"].sum()),
        "e3_negative_days": int(daily["e3_score"].lt(0).sum()),
        "e3_negative_daily_loss_top5_share": float(negative.nlargest(5).sum() / negative.sum()),
        "e3_negative_daily_loss_top10_share": float(negative.nlargest(10).sum() / negative.sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    paired.loc[:, [*KEYS, "taiex_return", "taiex_regime", "p_up", "e1_pred", "e3_pred",
                   "realized_return", "e1_contribution", "e3_contribution", "delta_contribution"]].to_csv(
        output_dir / "paired_rows.csv", index=False,
    )
    daily.to_csv(output_dir / "daily_scores.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(summary, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)
