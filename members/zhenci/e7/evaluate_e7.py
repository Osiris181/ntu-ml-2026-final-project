"""Evaluate train-only expected-payoff cutoffs on saved E1 probabilities.

From the repository root:
    uv run python -m members.zhenci.e7.evaluate_e7

No XGBoost model is trained, no 2026 outcomes are read, and no submission is
created. Payoff estimates are fitted on each fold's earlier training rows.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from members.zhenci.e1.save_e1 import _sha256
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e4.train_e4 import (
    E1_CONFIG_ID, E3_RULE_ID, align_saved_probabilities, checked_source_score,
    load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.splits import E1_SEEDS, FOLDS, split_market

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_E1_SUMMARY = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
DEFAULT_E1_PREDICTIONS = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"
DEFAULT_E3_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_xgboost_e7_payoff_v1"

PSEUDO_COUNT_PER_CLASS = 100
ARMS = (
    "e3_taiex_control",
    "e1_0500_control",
    "pooled_payoff",
    "regime_payoff",
    "stock_payoff_shrunk_100",
)


def _state(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("TAIEX feature contains infinity")
    return pd.Series(
        np.where(np.isnan(numeric), "missing", np.where(numeric < 0, "negative", "nonnegative")),
        index=values.index,
    )


def _payoff_cutoff(positive_mean: float, nonup_loss_mean: float) -> float:
    if not np.isfinite([positive_mean, nonup_loss_mean]).all() or positive_mean <= 0 or nonup_loss_mean < 0:
        raise ValueError("Invalid historical payoff means")
    denominator = positive_mean + nonup_loss_mean
    if denominator <= 0:
        raise ValueError("No positive historical payoff")
    return float(nonup_loss_mean / denominator)


def fit_payoff_cutoffs(train_rows: pd.DataFrame, train_taiex: pd.Series, fold_name: str) -> tuple[dict, pd.DataFrame]:
    """Fit three cutoff policies using realized returns from training dates only.

    The non-up class includes flat days, whose loss magnitude is zero. A
    stock's class-conditional payoff means each receive 100 pseudo-observations
    at the pooled mean; no validation or target-date outcomes enter fitting.
    """
    if train_rows.empty or len(train_rows) != len(train_taiex):
        raise ValueError("Training payoff rows and TAIEX values are misaligned")
    opened = train_rows["etl:adj_open"].to_numpy(dtype=float)
    closed = train_rows["etl:adj_close"].to_numpy(dtype=float)
    if not np.isfinite(opened).all() or not np.isfinite(closed).all() or (opened <= 0).any():
        raise ValueError("Historical adjusted prices are missing or invalid")
    ret = closed / opened - 1
    if not np.isfinite(ret).all():
        raise ValueError("Historical adjusted returns are nonfinite")
    positive = ret > 0
    if not positive.any() or positive.all():
        raise ValueError("Training data need both positive and non-up returns")
    gain = np.maximum(ret, 0)
    loss = np.maximum(-ret, 0)
    positive_mean = float(gain[positive].mean())
    nonup_loss_mean = float(loss[~positive].mean())
    pooled = _payoff_cutoff(positive_mean, nonup_loss_mean)

    work = pd.DataFrame({
        "stock_id": train_rows["stock_id"].to_numpy(),
        "state": _state(train_taiex).to_numpy(),
        "positive": positive, "gain": gain, "loss": loss,
    })
    grouped = work.groupby("stock_id", sort=True).agg(
        n=("positive", "size"), n_positive=("positive", "sum"),
        gain_sum=("gain", "sum"), loss_sum=("loss", "sum"),
    )
    grouped["n_nonup"] = grouped["n"] - grouped["n_positive"]
    k = PSEUDO_COUNT_PER_CLASS
    grouped["positive_mean"] = (grouped["gain_sum"] + k * positive_mean) / (grouped["n_positive"] + k)
    grouped["nonup_loss_mean"] = (grouped["loss_sum"] + k * nonup_loss_mean) / (grouped["n_nonup"] + k)
    grouped["cutoff"] = grouped["nonup_loss_mean"] / (grouped["positive_mean"] + grouped["nonup_loss_mean"])
    if not np.isfinite(grouped["cutoff"]).all() or not grouped["cutoff"].between(0, 1).all():
        raise ValueError("Invalid stock-level cutoff")

    regime = {}
    for state, rows in work.groupby("state", sort=True):
        up = rows["positive"].to_numpy(dtype=bool)
        if not up.any() or up.all():
            regime[state] = pooled
        else:
            regime[state] = _payoff_cutoff(float(rows.loc[up, "gain"].mean()),
                                           float(rows.loc[~up, "loss"].mean()))

    details = [{
        "fold": fold_name, "scope": "pooled", "key": "all_stocks",
        "n": len(work), "n_positive": int(positive.sum()), "n_nonup": int((~positive).sum()),
        "positive_mean": positive_mean, "nonup_loss_mean": nonup_loss_mean,
        "cutoff": pooled, "pseudo_count_per_class": 0,
    }]
    for stock_id, row in grouped.iterrows():
        details.append({
            "fold": fold_name, "scope": "stock", "key": stock_id,
            "n": int(row["n"]), "n_positive": int(row["n_positive"]),
            "n_nonup": int(row["n_nonup"]),
            "positive_mean": float(row["positive_mean"]),
            "nonup_loss_mean": float(row["nonup_loss_mean"]),
            "cutoff": float(row["cutoff"]), "pseudo_count_per_class": k,
        })
    for state, cutoff in regime.items():
        rows = work.loc[work.state.eq(state)]
        up = rows["positive"].to_numpy(dtype=bool)
        details.append({
            "fold": fold_name, "scope": "regime", "key": state,
            "n": len(rows), "n_positive": int(up.sum()), "n_nonup": int((~up).sum()),
            "positive_mean": float(rows.loc[up, "gain"].mean()) if up.any() else positive_mean,
            "nonup_loss_mean": float(rows.loc[~up, "loss"].mean()) if (~up).any() else nonup_loss_mean,
            "cutoff": cutoff, "pseudo_count_per_class": 0,
        })
    return {"pooled": pooled, "stock": grouped["cutoff"], "regime": regime}, pd.DataFrame(details)


def apply_payoff_cutoffs(
    p_up: np.ndarray, stock_ids: pd.Series, taiex: pd.Series,
    fitted: dict, arm: str, fixed_e3_rule: dict,
) -> np.ndarray:
    probabilities = np.asarray(p_up, dtype=float)
    if len(probabilities) != len(stock_ids) or len(probabilities) != len(taiex):
        raise ValueError("Probability and feature rows are misaligned")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid E1 probabilities")
    if arm == "e3_taiex_control":
        return regime_predictions(probabilities, taiex.to_numpy(dtype=float), fixed_e3_rule)
    if arm == "e1_0500_control":
        return (probabilities > 0.5).astype(int)
    if arm == "pooled_payoff":
        thresholds = np.full(len(probabilities), fitted["pooled"])
    elif arm == "regime_payoff":
        thresholds = _state(taiex).map(fitted["regime"]).fillna(fitted["pooled"]).to_numpy(dtype=float)
    elif arm == "stock_payoff_shrunk_100":
        thresholds = stock_ids.map(fitted["stock"]).to_numpy(dtype=float)
        if not np.isfinite(thresholds).all():
            raise ValueError("A target stock has no training payoff estimate")
    else:
        raise ValueError(f"Unknown E7 arm: {arm}")
    return (probabilities > thresholds).astype(int)


def run(
    data_dir: Path, f0_path: Path, e1_summary_path: Path,
    e1_predictions_path: Path, e3_summary_path: Path, output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    e1_summary = load_control_summary(e1_summary_path)
    fixed_rule, e3_summary = load_fixed_rule(e3_summary_path, E1_CONFIG_ID)
    e3_recorded = next(row for row in e3_summary["comparisons"] if row["config_id"] == E3_RULE_ID)
    saved = load_saved_e1_predictions(e1_predictions_path, e1_summary)
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    if len(f0) != len(market) or not f0.loc[:, KEYS].equals(market.loc[:, KEYS]):
        raise ValueError("E7 requires the original historical F0 and market row keys")
    taiex_f0 = f0.loc[:, [*KEYS, "taiex_return"]]

    scores = {arm: {seed: {} for seed in E1_SEEDS} for arm in ARMS}
    metrics = []
    cutoff_frames = []
    prediction_frames = []
    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        if set(train_rows.stock_id) != set(all_ids):
            raise ValueError(f"Training fold {fold.name} lacks some of the 100 stocks")
        train = train_rows.loc[:, KEYS].merge(taiex_f0, on=KEYS, how="left", sort=False,
                                             validate="one_to_one", indicator=True)
        valid = valid_rows.loc[:, KEYS].merge(taiex_f0, on=KEYS, how="left", sort=False,
                                             validate="one_to_one", indicator=True)
        if not train["_merge"].eq("both").all() or not valid["_merge"].eq("both").all():
            raise ValueError(f"Missing F0 TAIEX rows in fold {fold.name}")
        if not train.loc[:, KEYS].equals(train_rows.loc[:, KEYS]) or not valid.loc[:, KEYS].equals(valid_rows.loc[:, KEYS]):
            raise ValueError(f"F0 TAIEX join changed row order in fold {fold.name}")
        fitted, cutoff_details = fit_payoff_cutoffs(train_rows, train["taiex_return"], fold.name)
        cutoff_frames.append(cutoff_details)
        for seed in E1_SEEDS:
            p_up = align_saved_probabilities(valid_rows, saved, fold.name, seed)
            one_run = valid_rows.loc[:, KEYS].copy()
            one_run.insert(0, "seed", seed)
            one_run.insert(0, "fold", fold.name)
            one_run["p_up"] = p_up
            for arm in ARMS:
                labels = apply_payoff_cutoffs(p_up, valid["stock_id"], valid["taiex_return"],
                                              fitted, arm, fixed_rule)
                predictions = valid_rows.loc[:, KEYS].copy()
                predictions["predicted_y"] = labels
                result = evaluate_predictions(valid_rows, predictions)
                scores[arm][seed][fold.name] = result["official_score"]
                metrics.append({
                    "member": e1_summary["member"], "model": "XGBoost",
                    "experiment": "E1" if arm == "e1_0500_control" else "E2",
                    "config_id": arm, "fold": fold.name, "seed": seed,
                    **result, "train_seconds": None, "predict_seconds": None,
                    "predicted_up_rate": float(labels.mean()),
                })
                one_run[f"predicted_y_{arm}"] = labels
                if arm in ("e1_0500_control", "e3_taiex_control"):
                    recorded = e1_summary if arm == "e1_0500_control" else e3_recorded
                    checked_source_score(recorded["fold_official_scores_by_seed"], seed, fold.name,
                                         result["official_score"], "E1" if arm == "e1_0500_control" else "E3")
            prediction_frames.append(one_run)
        print(f"{fold.name}: fitted payoff cutoffs on earlier training rows; evaluated five seeds", flush=True)

    comparisons = [{
        "arm": arm, "fold_official_scores_by_seed": scores[arm], **e1_seed_summary(scores[arm]),
    } for arm in ARMS]
    selected = max(comparisons, key=lambda row: row["mean_cv_official_score"])
    incumbent = comparisons[0]
    selected_predictions = pd.concat(prediction_frames, ignore_index=True).loc[
        :, [*KEYS, "fold", "seed", "p_up", f"predicted_y_{selected['arm']}"]
    ].rename(columns={f"predicted_y_{selected['arm']}": "predicted_y"})
    selected_predictions.insert(0, "config_id", selected["arm"])
    selected_predictions.insert(0, "experiment", "E1" if selected["arm"] == "e1_0500_control" else "E2")
    selected_predictions.insert(0, "model", "XGBoost")
    selected_predictions.insert(0, "member", e1_summary["member"])

    summary = {
        "experiment_label": "E7 historical payoff decision rules",
        "experiment": "E2", "feature_set": "F0",
        "source_model_config_id": E1_CONFIG_ID,
        "pooled_formula": "threshold = mean loss on non-up days / (mean gain on up days + mean loss on non-up days)",
        "stock_shrinkage": f"{PSEUDO_COUNT_PER_CLASS} pooled pseudo-observations for each stock's up and non-up payoff mean",
        "regime_definition": "same F0 prior TAIEX return: negative, nonnegative, or missing",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "source_e1_summary": str(e1_summary_path), "source_e1_summary_sha256": _sha256(e1_summary_path),
        "source_e1_predictions": str(e1_predictions_path), "source_e1_predictions_sha256": _sha256(e1_predictions_path),
        "source_e3_summary": str(e3_summary_path), "source_e3_summary_sha256": _sha256(e3_summary_path),
        "source_f0": str(f0_path), "source_f0_sha256": _sha256(f0_path),
        "fixed_e3_rule": fixed_rule,
        "comparisons": comparisons,
        "incumbent_arm": incumbent["arm"],
        "incumbent_mean_cv_official_score": incumbent["mean_cv_official_score"],
        "selected_arm_by_cv": selected["arm"],
        "selected_mean_cv_official_score": selected["mean_cv_official_score"],
        "improvement_over_e3": selected["mean_cv_official_score"] - incumbent["mean_cv_official_score"],
        "payoff_rule_beats_e3": selected["arm"] not in ("e1_0500_control", "e3_taiex_control") and selected["mean_cv_official_score"] > incumbent["mean_cv_official_score"],
        "selection_method": "Mean of three target-stock year official scores within each seed, then mean of five seeds; incumbent first on ties",
        "exploratory_note": "The payoff policies were inspected on these same CV years before this script was written; the result is a reproducible exploratory comparison, not independent confirmation",
        "status": "validation only; no new XGBoost fit, deployment artifact, or submission",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics).loc[:, [*METRIC_FIELDS, "predicted_up_rate"]].to_csv(
        output_dir / "fold_metrics.csv", index=False,
    )
    pd.concat(cutoff_frames, ignore_index=True).to_csv(output_dir / "payoff_cutoffs.csv", index=False)
    selected_predictions.loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "selected_validation_predictions.csv", index=False,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    for comparison in comparisons:
        print(f"{comparison['arm']}: mean CV official score {comparison['mean_cv_official_score']:.8f}")
    print(f"Selected {selected['arm']}; payoff rule beats E3: {summary['payoff_rule_beats_e3']}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_E1_SUMMARY)
    parser.add_argument("--e1-predictions", type=Path, default=DEFAULT_E1_PREDICTIONS)
    parser.add_argument("--e3-summary", type=Path, default=DEFAULT_E3_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.e1_summary, args.e1_predictions, args.e3_summary, args.output_dir)
