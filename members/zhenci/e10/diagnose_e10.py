"""Read-only E10 diagnostic of saved E3 probabilities, F0 shift and scoring.

Run from final_project: uv run python -m members.zhenci.e10.diagnose_e10
No training, fitting, threshold selection, or existing-result overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import FOLDS, HOLDOUT, split_market
from shared.updated_features import load_updated_market

OUTPUT = Path(__file__).resolve().parent / "outputs"
PERIODS = ("2023", "2024", "2025", "holdout_jan_may_2026",
           "phase1_jun_sep_2026", "oct_2026_01_22")
FROZEN_PERIODS = PERIODS[3:]
OPEN, CLOSE = "etl:adj_open", "etl:adj_close"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_model(artifact: Path, f0_original: Path) -> tuple[xgb.Booster, BaselinePreprocessor, dict]:
    with (artifact / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("train_end") != "2025-12-31"
            or manifest.get("training_rows") != 145847
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("threshold") != .5
            or _sha(f0_original) != manifest.get("f0_csv_sha256")):
        raise ValueError("Frozen model training contract changed")
    for name, field in (("model.json", "model_sha256"),
                        ("preprocessor.json", "preprocessor_sha256"),
                        ("feature_definitions.py", "feature_definitions_sha256"),
                        ("preprocessing_definitions.py", "preprocessing_definitions_sha256")):
        if _sha(artifact / name) != manifest.get(field):
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    if (_sha(PROJECT / "shared/features.py") != manifest["feature_definitions_sha256"]
            or _sha(PROJECT / "shared/preprocessing.py")
            != manifest["preprocessing_definitions_sha256"]):
        raise ValueError("Shared F0/preprocessing changed from frozen artifact")
    with (artifact / "preprocessor.json").open(encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    return xgb.Booster(model_file=str(artifact / "model.json")), preprocessor, manifest


def _finish(period: str, evaluate: pd.DataFrame, probabilities: pd.DataFrame,
            f0: pd.DataFrame, rule: dict, expected_e3: pd.Series | None = None) -> pd.DataFrame:
    validate_keys(evaluate)
    validate_keys(probabilities)
    if len(evaluate) != len(probabilities):
        raise ValueError(f"Missing probability rows: {period}")
    merged = evaluate[KEYS + [OPEN, CLOSE]].merge(
        probabilities[KEYS + ["p_up"]], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Probability keys differ: {period}")
    merged = merged.drop(columns="_merge").merge(
        f0[KEYS + list(NUMERIC_FEATURES) + ["day_of_week"]],
        on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"F0 keys differ: {period}")
    merged = merged.drop(columns="_merge")
    if (merged[OPEN].le(0).any()
            or merged[[OPEN, CLOSE, "p_up"]].isna().any().any()
            or not np.isfinite(merged[[OPEN, CLOSE, "p_up"]].to_numpy()).all()
            or not merged["p_up"].between(0, 1).all()):
        raise ValueError(f"Invalid prices or probabilities: {period}")
    merged["e3_pred"] = regime_predictions(
        merged["p_up"].to_numpy(), merged["taiex_return"].to_numpy(), rule)
    if expected_e3 is not None:
        expected = probabilities[KEYS].copy()
        expected["saved_e3"] = expected_e3.to_numpy()
        checked = merged[KEYS + ["e3_pred"]].merge(expected, on=KEYS, validate="one_to_one")
        if not checked["e3_pred"].eq(checked["saved_e3"]).all():
            raise ValueError(f"Reconstructed E3 differs from saved predictions: {period}")
    merged["realized_return"] = (merged[CLOSE] - merged[OPEN]) / merged[OPEN]
    merged["actual_up"] = merged["realized_return"].gt(0).astype(int)
    merged["contribution"] = (2 * merged["e3_pred"] - 1) * merged["realized_return"]
    official = evaluate_predictions(
        evaluate, merged[KEYS + ["e3_pred"]].rename(columns={"e3_pred": "predicted_y"}))
    if not np.isclose(official["official_score"], merged["contribution"].mean(),
                      rtol=0, atol=1e-14):
        raise ValueError(f"Official score decomposition failed: {period}")
    merged.insert(0, "period", period)
    return merged.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, xgb.Booster, dict]:
    target_ids, all_ids = load_metadata(PROJECT / "data")
    market = load_market(PROJECT / "data")
    f0_path = PROJECT / "data/processed/f0.csv"
    f0 = load_f0_dataset(f0_path)
    updated_f0 = load_f0_dataset(PROJECT / "data/processed/f0_with_updates.csv")
    artifact = PROJECT / "results/zhenci_e1_full_pre_holdout_artifact"
    booster, preprocessor, manifest = _validate_model(artifact, f0_path)
    with (PROJECT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json").open(
        encoding="utf-8") as file:
        e3_summary = json.load(file)
        rule = selected_rule(e3_summary, manifest["config_id"])

    source = pd.read_csv(PROJECT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv",
                         dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    expected_seeds = {42, 64, 128, 256, 512}
    if (set(source["seed"]) != expected_seeds
            or set(source["config_id"]) != {manifest["config_id"]}
            or set(source["fold"]) != {fold.name for fold in FOLDS}
            or source.groupby(["fold", *KEYS])["seed"].nunique().ne(5).any()):
        raise ValueError("Saved CV configuration or five-seed coverage changed")
    if source.duplicated(["fold", "seed", *KEYS]).any():
        raise ValueError("Duplicate CV prediction keys")
    if source.groupby(["fold", *KEYS])["p_up"].nunique().ne(1).any():
        raise ValueError("Saved CV seeds differ; do not select one seed")
    source = source.loc[source["seed"].eq(42)]
    frames = {}
    for fold in FOLDS:
        train, evaluate = split_market(market, target_ids, fold)
        if train["date"].max() >= evaluate["date"].min():
            raise ValueError("Chronological split violated")
        pred = source.loc[source["fold"].eq(fold.name)]
        frames[fold.name] = _finish(fold.name, evaluate, pred, f0, rule)
    cv_mean = float(np.mean([frames[fold.name]["contribution"].mean() for fold in FOLDS]))
    if not np.isclose(cv_mean, e3_summary["selected_same_period_cv"]["mean_cv_official_score"],
                      rtol=0, atol=1e-12):
        raise ValueError("Reconstructed E3 CV score differs from archived result")
    _, evaluate = split_market(market, target_ids, HOLDOUT)
    holdout = pd.read_csv(PROJECT / "results/zhenci_xgboost_e7_holdout_v1/holdout_predictions.csv",
                          dtype={"stock_id": str}, parse_dates=["date"])
    frames["holdout_jan_may_2026"] = _finish(
        "holdout_jan_may_2026", evaluate, holdout, f0, rule,
        holdout["predicted_y_e3_taiex_control"])

    updated = load_updated_market(PROJECT / "updated_data_7_Nov/data", all_ids)
    if updated["date"].min() <= market["date"].max():
        raise ValueError("Updated market overlaps original history")
    full_market = pd.concat([market.drop(columns="y"), updated], ignore_index=True)
    phase_saved = pd.read_csv(
        PROJECT / "members/zhenci/diagnostics/e1_e3_gap_outputs/paired_rows.csv",
        dtype={"stock_id": str}, parse_dates=["date"])
    phase_eval = phase_saved[KEYS].merge(full_market[KEYS + [OPEN, CLOSE]],
                                         on=KEYS, validate="one_to_one", how="left")
    if len(phase_eval) != 4200:
        raise ValueError("Phase 1 key count changed")
    frames["phase1_jun_sep_2026"] = _finish(
        "phase1_jun_sep_2026", phase_eval, phase_saved, updated_f0, rule,
        phase_saved["e3_pred"])
    if not np.allclose(frames["phase1_jun_sep_2026"]["realized_return"],
                       phase_saved.sort_values(["date", "stock_id"])["realized_return"],
                       rtol=1e-10, atol=1e-12):
        raise ValueError("Phase 1 adjusted returns differ from archived audit")

    oct_eval = updated.loc[updated["stock_id"].isin(target_ids)
                           & updated["date"].between("2026-10-01", "2026-10-22"),
                           KEYS + [OPEN, CLOSE]].copy()
    if (len(oct_eval) != 750 or oct_eval["date"].nunique() != 15
            or not oct_eval.groupby("date").size().eq(50).all()):
        raise ValueError("October 1-22 window is incomplete")
    oct_f0 = oct_eval[KEYS].merge(updated_f0, on=KEYS, validate="one_to_one", how="left")
    transformed = preprocessor.transform(oct_f0[list(F0_FEATURES)])
    if transformed.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("Frozen transformed feature order changed")
    oct_prob = oct_f0[KEYS].copy()
    oct_prob["p_up"] = booster.predict(xgb.DMatrix(transformed))
    frames["oct_2026_01_22"] = _finish(
        "oct_2026_01_22", oct_eval, oct_prob, updated_f0, rule)
    old_oct = pd.read_csv(PROJECT / "members/zhenci/e8/outputs/reserved_metrics.csv")
    old_e3 = old_oct.loc[old_oct["approach"].eq("e3"), "official_score"].iat[0]
    if not np.isclose(old_e3, frames["oct_2026_01_22"]["contribution"].mean(),
                      rtol=0, atol=1e-12):
        raise ValueError("October frozen-model score differs from E8 audit")
    return frames, market, f0, booster, rule


def _period_summary(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, months, days = [], [], []
    for period, frame in frames.items():
        def row(group: pd.DataFrame) -> dict:
            return {"n": len(group), "trading_days": group["date"].nunique(),
                    "predicted_up_rate": float(group["e3_pred"].mean()),
                    "actual_up_rate": float(group["actual_up"].mean()),
                    "mean_probability": float(group["p_up"].mean()),
                    "median_probability": float(group["p_up"].median()),
                    "p10_probability": float(group["p_up"].quantile(.1)),
                    "p90_probability": float(group["p_up"].quantile(.9)),
                    "mean_intraday_return": float(group["realized_return"].mean()),
                    "median_intraday_return": float(group["realized_return"].median()),
                    "official_score": float(group["contribution"].mean()),
                    "brier_score": float(np.mean((group["p_up"] - group["actual_up"])**2))}
        summary.append({"period": period, **row(frame)})
        for month, group in frame.groupby(frame["date"].dt.to_period("M"), sort=True):
            months.append({"period": period, "month": str(month), **row(group)})
        for day, group in frame.groupby("date", sort=True):
            days.append({"period": period, "date": day, **row(group)})
    return pd.DataFrame(summary), pd.DataFrame(months), pd.DataFrame(days)


def _calibration(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins, metrics, outcomes = [], [], []
    for period, frame in frames.items():
        x = frame.copy()
        x["bin"] = pd.cut(x["p_up"], np.linspace(0, 1, 11),
                          labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
                          include_lowest=True)
        if x["bin"].isna().any():
            raise ValueError("Probability outside [0,1]")
        ece = 0.0
        for label, group in x.groupby("bin", observed=True, sort=True):
            mean_p = float(group["p_up"].mean())
            freq = float(group["actual_up"].mean())
            ece += (len(group) / len(x)) * abs(mean_p - freq)
            bins.append({"period": period, "probability_bin": str(label), "n": len(group),
                         "mean_probability": mean_p, "observed_up_rate": freq,
                         "mean_realized_return": float(group["realized_return"].mean()),
                         "e3_up_rate": float(group["e3_pred"].mean()),
                         "official_score": float(group["contribution"].mean())})
        metrics.append({"period": period, "brier_score":
                        float(np.mean((x["p_up"] - x["actual_up"])**2)),
                        "ece_fixed_tenths": ece})
        x["decision_outcome"] = np.select(
            [x["e3_pred"].eq(1) & x["actual_up"].eq(1),
             x["e3_pred"].eq(0) & x["actual_up"].eq(0),
             x["e3_pred"].eq(1) & x["actual_up"].eq(0)],
            ["correct_up", "correct_down", "incorrect_up"], default="incorrect_down")
        for label, group in x.groupby("decision_outcome", sort=True):
            outcomes.append({"period": period, "decision_outcome": label, "n": len(group),
                             "fraction": len(group) / len(x),
                             "mean_realized_return": float(group["realized_return"].mean()),
                             "mean_probability": float(group["p_up"].mean()),
                             "sum_contribution": float(group["contribution"].sum()),
                             "contribution_to_period_score": float(group["contribution"].sum()/len(x))})
    return pd.DataFrame(bins), pd.DataFrame(metrics), pd.DataFrame(outcomes)


def _threshold_conditions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for period, frame in frames.items():
        x = frame.copy()
        x["taiex_condition"] = np.select(
            [x["taiex_return"].isna(), x["taiex_return"].lt(0)],
            ["missing", "negative"], default="nonnegative")
        x["e1_pred"] = x["p_up"].gt(.5).astype(int)
        x["e1_contribution"] = (2*x["e1_pred"]-1)*x["realized_return"]
        for condition, group in [("all", x), *x.groupby("taiex_condition", sort=True)]:
            rows.append({"period": period, "taiex_condition": condition,
                         "n": len(group), "days": group["date"].nunique(),
                         "mean_probability": float(group["p_up"].mean()),
                         "e1_up_rate": float(group["e1_pred"].mean()),
                         "e3_up_rate": float(group["e3_pred"].mean()),
                         "actual_up_rate": float(group["actual_up"].mean()),
                         "mean_intraday_return": float(group["realized_return"].mean()),
                         "e1_official_score": float(group["e1_contribution"].mean()),
                         "e3_official_score": float(group["contribution"].mean()),
                         "e1_e3_disagreements": int(group["e1_pred"].ne(group["e3_pred"]).sum())})
    return pd.DataFrame(rows)


def _monthly_feature_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for period in FROZEN_PERIODS:
        frame = frames[period]
        for month, group in frame.groupby(frame["date"].dt.to_period("M"), sort=True):
            for feature in NUMERIC_FEATURES:
                values = group[feature].dropna()
                rows.append({"period": period, "month": str(month), "feature": feature,
                             "rows": len(group), "missing_rate": float(group[feature].isna().mean()),
                             "mean": float(values.mean()) if len(values) else np.nan,
                             "median": float(values.median()) if len(values) else np.nan,
                             "q10": float(values.quantile(.1)) if len(values) else np.nan,
                             "q90": float(values.quantile(.9)) if len(values) else np.nan})
    return pd.DataFrame(rows)


def _psi(ref: np.ndarray, later: np.ndarray) -> float:
    if len(ref) < 100 or len(later) < 100:
        return float("nan")
    boundaries = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    if len(boundaries) < 2:
        return float("nan")
    a = np.bincount(np.searchsorted(boundaries, ref, side="right"), minlength=len(boundaries)+1)
    b = np.bincount(np.searchsorted(boundaries, later, side="right"), minlength=len(boundaries)+1)
    p = (a + .5) / (len(ref) + .5 * len(a))
    q = (b + .5) / (len(later) + .5 * len(b))
    return float(np.sum((q-p) * np.log(q/p)))


def _feature_shift(frames: dict[str, pd.DataFrame], market: pd.DataFrame,
                   f0: pd.DataFrame, booster: xgb.Booster) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_ids, _ = load_metadata(PROJECT / "data")
    importance = booster.get_score(importance_type="total_gain")
    grouped_gain: dict[str, float] = {feature: 0.0 for feature in F0_FEATURES}
    for transformed, value in importance.items():
        logical = transformed.split("=", 1)[0]
        if logical not in grouped_gain:
            raise ValueError(f"Unexpected trained feature: {transformed}")
        grouped_gain[logical] += float(value)
    total_gain = sum(grouped_gain.values())
    gain = pd.DataFrame({"feature": list(grouped_gain),
                         "total_gain": list(grouped_gain.values())})
    gain["share_of_total_gain"] = gain["total_gain"] / total_gain
    rank = {name: i + 1 for i, name in enumerate(gain.sort_values("total_gain", ascending=False)["feature"])}
    gain["rank"] = gain["feature"].map(rank)
    shifts, stock = [], []
    for period, later in frames.items():
        train_end = {"2023": "2022-12-31", "2024": "2023-12-31",
                     "2025": "2024-12-31"}.get(period, "2025-12-31")
        train_keys = market.loc[market["stock_id"].isin(target_ids)
                                & market["date"].between("2020-01-01", train_end), KEYS]
        ref = train_keys.merge(f0, on=KEYS, validate="one_to_one")
        for feature in NUMERIC_FEATURES:
            a = ref[feature].dropna().to_numpy(dtype=float)
            b = later[feature].dropna().to_numpy(dtype=float)
            if not len(a) or not len(b):
                continue
            q10a, q25a, ma, q75a, q90a = np.quantile(a, [.1,.25,.5,.75,.9])
            q10b, q25b, mb, q75b, q90b = np.quantile(b, [.1,.25,.5,.75,.9])
            iqr = q75a-q25a
            shifts.append({"period": period, "feature": feature,
                           "training_target50_rows": len(ref), "evaluation_rows": len(later),
                           "train_mean": float(np.mean(a)), "eval_mean": float(np.mean(b)),
                           "train_median": float(ma), "eval_median": float(mb),
                           "train_q10": float(q10a), "eval_q10": float(q10b),
                           "train_q90": float(q90a), "eval_q90": float(q90b),
                           "train_iqr": float(iqr), "eval_iqr": float(q75b-q25b),
                           "train_missing_rate": float(ref[feature].isna().mean()),
                           "eval_missing_rate": float(later[feature].isna().mean()),
                           "abs_median_shift_in_train_iqr": float(abs(mb-ma)/iqr) if iqr>1e-12 else np.nan,
                           "psi_train_deciles": _psi(a,b),
                           "frozen_model_gain_rank": rank[feature],
                           "frozen_model_gain_share": grouped_gain[feature]/total_gain})
        for stock_id, group in later.groupby("stock_id", sort=True):
            prior = ref.loc[ref["stock_id"].eq(stock_id)]
            stock.append({"period": period, "stock_id": stock_id, "n": len(group),
                          "mean_prior_intraday_return": float(group["intraday_return_1d"].mean()),
                          "train_mean_prior_intraday_return": float(prior["intraday_return_1d"].mean()),
                          "median_relative_volume": float(group["relative_volume_20d"].median()),
                          "train_median_relative_volume": float(prior["relative_volume_20d"].median()),
                          "e3_up_rate": float(group["e3_pred"].mean()),
                          "actual_up_rate": float(group["actual_up"].mean()),
                          "official_score": float(group["contribution"].mean())})
    return pd.DataFrame(shifts), pd.DataFrame(stock), gain.sort_values("rank")


def _reference_checks(frames: dict[str, pd.DataFrame], market: pd.DataFrame,
                      f0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check both categorical features and the complete 100-stock training universe."""
    target_ids, _ = load_metadata(PROJECT / "data")
    categories, all_stock_shifts = [], []
    for period, later in frames.items():
        train_end = {"2023": "2022-12-31", "2024": "2023-12-31",
                     "2025": "2024-12-31"}.get(period, "2025-12-31")
        all_keys = market.loc[market["date"].between("2020-01-01", train_end), KEYS]
        all_ref = all_keys.merge(f0, on=KEYS, validate="one_to_one")
        target_ref = all_ref.loc[all_ref["stock_id"].isin(target_ids)]
        for feature in ("stock_id", "day_of_week"):
            a = target_ref[feature].astype(str).value_counts(normalize=True)
            b = later[feature].astype(str).value_counts(normalize=True)
            labels = a.index.union(b.index)
            categories.append({"period": period, "feature": feature,
                               "reference": "target_50_stocks",
                               "train_category_count": len(a),
                               "eval_category_count": len(b),
                               "unseen_eval_fraction": float(b.loc[b.index.difference(a.index)].sum()),
                               "total_variation_distance": float((a.reindex(labels, fill_value=0)
                                                                 - b.reindex(labels, fill_value=0)).abs().sum()/2)})
        for feature in NUMERIC_FEATURES:
            a = all_ref[feature].dropna().to_numpy(dtype=float)
            b = later[feature].dropna().to_numpy(dtype=float)
            if not len(a) or not len(b):
                continue
            all_stock_shifts.append({"period": period, "feature": feature,
                                     "train_all100_rows": len(all_ref), "evaluation_rows": len(later),
                                     "train_median": float(np.median(a)),
                                     "eval_median": float(np.median(b)),
                                     "train_missing_rate": float(all_ref[feature].isna().mean()),
                                     "eval_missing_rate": float(later[feature].isna().mean()),
                                     "psi_train_deciles": _psi(a, b)})
    return pd.DataFrame(categories), pd.DataFrame(all_stock_shifts)


def _svg_line_monthly(months: pd.DataFrame, path: Path) -> None:
    df = months.loc[months["period"].isin(FROZEN_PERIODS)].sort_values("month")
    # Each month appears exactly once in the frozen-model sequence.
    df = df.drop_duplicates("month")
    width, height, left, right, top, bottom = 900, 380, 65, 25, 35, 85
    xs = np.linspace(left, width-right, len(df))
    ys = lambda v: top + (1-v) * (height-top-bottom)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="65" y="22" font-size="17" font-family="Arial">Frozen model: predicted up vs actual up by month</text>']
    for value in np.arange(0, 1.01, .2):
        y = ys(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        elements.append(f'<text x="12" y="{y+4:.1f}" font-size="12" font-family="Arial">{value:.0%}</text>')
    for field, color, label in (("predicted_up_rate", "#c24c32", "E3 predicted up"),
                                ("actual_up_rate", "#2463a1", "Actually up")):
        points = " ".join(f"{x:.1f},{ys(y):.1f}" for x,y in zip(xs, df[field]))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x,y in zip(xs, df[field]):
            elements.append(f'<circle cx="{x:.1f}" cy="{ys(y):.1f}" r="3" fill="{color}"/>')
    for x,label in zip(xs, df["month"]):
        elements.append(f'<text x="{x:.1f}" y="{height-58}" transform="rotate(50 {x:.1f} {height-58})" font-size="11" font-family="Arial">{escape(label)}</text>')
    elements += ['<line x1="590" y1="24" x2="615" y2="24" stroke="#c24c32" stroke-width="3"/>',
                 '<text x="620" y="28" font-size="12" font-family="Arial">E3 predicted up</text>',
                 '<line x1="750" y1="24" x2="775" y2="24" stroke="#2463a1" stroke-width="3"/>',
                 '<text x="780" y="28" font-size="12" font-family="Arial">Actually up</text>', '</svg>']
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_calibration(bins: pd.DataFrame, path: Path) -> None:
    width, height, left, top, size = 620, 560, 70, 55, 420
    palette = {"holdout_jan_may_2026":"#2071a7", "phase1_jun_sep_2026":"#c24c32", "oct_2026_01_22":"#8059a6"}
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="70" y="26" font-size="17" font-family="Arial">Frozen-model direction calibration</text>']
    for value in np.arange(0,1.01,.2):
        x=left+value*size; y=top+(1-value)*size
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+size}" stroke="#eeeeee"/>')
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+size}" y2="{y:.1f}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{x-10:.1f}" y="{top+size+18}" font-size="11" font-family="Arial">{value:.1f}</text>')
        elements.append(f'<text x="{left-30}" y="{y+4:.1f}" font-size="11" font-family="Arial">{value:.1f}</text>')
    elements.append(f'<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top}" stroke="#777" stroke-dasharray="5 4"/>')
    for i,(period,color) in enumerate(palette.items()):
        group=bins.loc[bins["period"].eq(period)].sort_values("mean_probability")
        for _,row in group.iterrows():
            x=left+row["mean_probability"]*size; y=top+(1-row["observed_up_rate"])*size
            radius=min(11,max(3,np.sqrt(row["n"])/3))
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72"/>')
        elements.append(f'<circle cx="{left+10+i*145}" cy="525" r="5" fill="{color}"/>')
        elements.append(f'<text x="{left+20+i*145}" y="529" font-size="11" font-family="Arial">{escape(period.replace("_2026", ""))}</text>')
    elements.append('<text x="195" y="504" font-size="12" font-family="Arial">Mean predicted probability</text>')
    elements.append('<text x="16" y="330" transform="rotate(-90 16 330)" font-size="12" font-family="Arial">Observed up fraction</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_shift(shifts: pd.DataFrame, path: Path) -> None:
    df = shifts.loc[shifts["period"].eq("phase1_jun_sep_2026")].sort_values(
        "abs_median_shift_in_train_iqr", ascending=False).head(10)
    width,height,left,top=1050,430,220,45
    scale=(width-left-200)/max(df["abs_median_shift_in_train_iqr"].max(),1e-9)
    elements=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="20" y="24" font-size="17" font-family="Arial">Phase 1 vs 2020-2025 target-stock F0 medians</text>']
    for i,(_,row) in enumerate(df.iterrows()):
        y=top+i*35; value=row["abs_median_shift_in_train_iqr"]
        elements.append(f'<text x="12" y="{y+18}" font-size="12" font-family="Arial">{escape(row["feature"])}</text>')
        elements.append(f'<rect x="{left}" y="{y+4}" width="{value*scale:.1f}" height="22" fill="#4d78a8"/>')
        elements.append(f'<text x="{left+value*scale+6:.1f}" y="{y+20}" font-size="11" font-family="Arial">{value:.2f} IQR; gain rank {int(row["frozen_model_gain_rank"])}</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_daily_scatter(days: pd.DataFrame, path: Path) -> None:
    df = days.loc[days["period"].isin(FROZEN_PERIODS)].copy()
    width, height, left, top, plot_w, plot_h = 820, 500, 80, 45, 620, 370
    xmin, xmax, ymin, ymax = -.04, .04, -.04, .04
    palette = {"holdout_jan_may_2026":"#2071a7", "phase1_jun_sep_2026":"#c24c32",
               "oct_2026_01_22":"#8059a6"}
    xp = lambda v: left + (v-xmin)/(xmax-xmin)*plot_w
    yp = lambda v: top + (ymax-v)/(ymax-ymin)*plot_h
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="80" y="25" font-size="17" font-family="Arial">Frozen E3: daily market return vs official score</text>']
    for v in np.arange(-.04, .041, .02):
        elements.append(f'<line x1="{xp(v):.1f}" y1="{top}" x2="{xp(v):.1f}" y2="{top+plot_h}" stroke="#eeeeee"/>')
        elements.append(f'<line x1="{left}" y1="{yp(v):.1f}" x2="{left+plot_w}" y2="{yp(v):.1f}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{xp(v)-15:.1f}" y="{top+plot_h+17}" font-size="11" font-family="Arial">{v:.0%}</text>')
        elements.append(f'<text x="{left-48}" y="{yp(v)+4:.1f}" font-size="11" font-family="Arial">{v:.0%}</text>')
    for _, row in df.iterrows():
        x, y = xp(row["mean_intraday_return"]), yp(row["official_score"])
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{palette[row["period"]]}" fill-opacity="0.7"/>')
    elements.append('<text x="260" y="452" font-size="12" font-family="Arial">Daily mean intraday return</text>')
    elements.append('<text x="17" y="300" transform="rotate(-90 17 300)" font-size="12" font-family="Arial">Daily official score</text>')
    for i, (period, color) in enumerate(palette.items()):
        elements.append(f'<circle cx="{80+i*230}" cy="476" r="5" fill="{color}"/>')
        elements.append(f'<text x="{92+i*230}" y="480" font-size="11" font-family="Arial">{escape(period)}</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def run(output_dir: Path = OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E10 output directory: {output_dir}")
    frames, market, f0, booster, rule = _data()
    periods, months, days = _period_summary(frames)
    bins, calibration, outcomes = _calibration(frames)
    threshold_conditions = _threshold_conditions(frames)
    monthly_features = _monthly_feature_summary(frames)
    shifts, stocks, gain = _feature_shift(frames, market, f0, booster)
    categories, all_stock_shifts = _reference_checks(frames, market, f0)
    # Global-market features repeat for every stock; a separate once-per-day view avoids
    # treating 50 copies as 50 independent global observations.
    global_daily = pd.concat([
        frame.groupby("date", sort=True)[["vix_normalized", "nasdaq_return",
                                           "sp500_return", "taiex_return"]].first().reset_index().assign(period=period)
        for period, frame in frames.items()], ignore_index=True)
    highest_positive_days = days.loc[days["mean_intraday_return"].gt(0)].sort_values(
        ["period", "mean_intraday_return"], ascending=[True,False]).groupby("period").head(10)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("period_summary",periods),("monthly_summary",months),("daily_summary",days),
                        ("probability_bins",bins),("calibration_metrics",calibration),
                        ("threshold_conditions",threshold_conditions),
                        ("monthly_feature_summary",monthly_features),
                        ("decision_outcomes",outcomes),("feature_shift",shifts),
                        ("categorical_shift",categories),("training_all100_feature_shift",all_stock_shifts),
                        ("stock_shift",stocks),("frozen_model_gain",gain),
                        ("global_daily",global_daily),("top_positive_days",highest_positive_days)):
        table.to_csv(output_dir/f"{name}.csv", index=False)
    _svg_line_monthly(months, output_dir/"up_rates.svg")
    _svg_calibration(bins, output_dir/"calibration.svg")
    _svg_shift(shifts, output_dir/"feature_shift.svg")
    _svg_daily_scatter(days, output_dir/"daily_score.svg")
    with (output_dir/"manifest.json").open("w",encoding="utf-8") as file:
        json.dump({"periods": PERIODS, "frozen_periods": FROZEN_PERIODS,
                   "excluded_invalid_date": "2026-10-23 and subsequent October/November dates",
                   "e3_rule": rule,
                   "input_sha256": {
                       "cv_probabilities": _sha(PROJECT/"results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"),
                       "holdout_probabilities": _sha(PROJECT/"results/zhenci_xgboost_e7_holdout_v1/holdout_predictions.csv"),
                       "phase1_probabilities": _sha(PROJECT/"members/zhenci/diagnostics/e1_e3_gap_outputs/paired_rows.csv"),
                       "original_f0": _sha(PROJECT/"data/processed/f0.csv"),
                       "updated_f0": _sha(PROJECT/"data/processed/f0_with_updates.csv"),
                       "frozen_artifact": _sha(PROJECT/"results/zhenci_e1_full_pre_holdout_artifact/manifest.json")}},
                  file, indent=2)
        file.write("\n")
    print(periods.to_string(index=False))
    return output_dir


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    run(args.output_dir)
