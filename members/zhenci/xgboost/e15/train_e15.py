"""Run E15 on 2023–2025 only: uv run python -m members.zhenci.xgboost.e15.train_e15."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e1.save_e1 import _sha256
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY,
    align_saved_probabilities, load_control_summary, load_fixed_rule,
    load_saved_e1_predictions,
)
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import CLOSE, OPEN, cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, NUMERIC_FEATURES, BaselinePreprocessor
from shared.splits import FOLDS, SEED, evaluation_keys, split_market

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
DEFAULT_F0 = ROOT / "data/processed/f0.csv"
ARM_NAMES = ("E3", "E15-A", "E15-B")
GLOBAL_FIELDS = ("vix_normalized", "nasdaq_return", "sp500_return", "taiex_return")
STOCK_FIELDS = [field for field in NUMERIC_FEATURES if field not in GLOBAL_FIELDS]
RANK_NAMES = {field: f"rank_{field}" for field in STOCK_FIELDS}
BLOCK_DAYS = 5
DRAWS = 3000


def rank_panel(f0: pd.DataFrame, reference_ids: list[str], target_ids: list[str]
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ranks and coverage from all supplied eligible Top100 snapshots.

    Call before selecting target rows. Absent peers are counted in coverage;
    the rank denominator counts only valid feature values for that date.
    """
    validate_keys(f0)
    reference = set(reference_ids)
    targets = set(target_ids)
    if (len(reference_ids) != 100 or len(reference) != 100
            or len(target_ids) != 50 or len(targets) != 50 or not targets <= reference):
        raise ValueError("Expected metadata Top100 and Top50 universes")
    if f0.empty or not set(f0["stock_id"]) <= reference:
        raise ValueError("F0 must contain snapshot rows from the fixed Top100 universe")
    peer_counts = f0["stock_id"].isin(reference - targets).groupby(f0["date"]).sum()
    if peer_counts.eq(0).any():
        raise ValueError("No non-target peer snapshots on at least one date; check the Top100 input")
    values = f0[STOCK_FIELDS].replace([np.inf, -np.inf], np.nan)
    by_date = values.groupby(f0["date"])
    counts = by_date.transform("count")
    ranks = (by_date.rank(method="average") - 1).div((counts - 1).where(counts.gt(1)))
    ranks = ranks.mask(values.notna() & counts.eq(1), .5)
    if (not ranks.isna().equals(values.isna())
            or ((ranks.lt(0) | ranks.gt(1)).any().any())):
        raise ValueError("Rank range or missingness changed")
    daily_counts = by_date.count().rename_axis(columns="feature")
    coverage = daily_counts.stack().rename("eligible_values").reset_index()
    constants = (by_date.nunique().eq(1) & daily_counts.ge(2))
    coverage["constant_group"] = constants.stack().reindex(
        coverage.set_index(["date", "feature"]).index).to_numpy()
    snapshots = f0.groupby("date").size()
    coverage["snapshot_rows"] = coverage["date"].map(snapshots)
    coverage["eligible_fraction_of_top100"] = coverage["eligible_values"] / 100
    return pd.concat([f0, ranks.rename(columns=RANK_NAMES)], axis=1), coverage


def transformed_inputs(train: pd.DataFrame, valid: pd.DataFrame
                       ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Reuse the shared training-only median, scaling, and category policy."""
    baseline = BaselinePreprocessor().fit(train)
    renamed = {rank: field for field, rank in RANK_NAMES.items()}

    def replace_stock_fields(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=STOCK_FIELDS).rename(columns=renamed)

    rank_train, rank_valid = map(replace_stock_fields, (train, valid))
    rank_processor = BaselinePreprocessor().fit(rank_train)
    original = (baseline.transform(train), baseline.transform(valid))
    replacement = tuple(rank_processor.transform(frame).rename(columns=RANK_NAMES)
                        for frame in (rank_train, rank_valid))
    rank_columns = list(RANK_NAMES.values())
    combined = []
    for unchanged, replaced in zip(original, replacement):
        retained = unchanged.columns.difference(STOCK_FIELDS, sort=False)
        if not unchanged[retained].equals(replaced[retained]):
            raise ValueError("Preprocessing changed retained globals or categories")
        combined.append(pd.concat([unchanged, replaced[rank_columns]], axis=1))
    return dict(zip(ARM_NAMES, (original, replacement, tuple(combined))))


def daily_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    daily = []
    for (fold, arm, date), group in rows.groupby(["fold", "arm", "date"], sort=True):
        # Pearson correlation of average ranks is Spearman. Constant groups are undefined.
        ranks = group[["p_up", "realized_return"]].rank(method="average")
        correlation = ranks.corr().iloc[0, 1]
        daily.append({"fold": fold, "arm": arm, "date": date, "n": len(group),
                      "contribution_sum": group["contribution"].sum(),
                      "daily_score": group["contribution"].mean(), "spearman": correlation})
    return pd.DataFrame(daily)


def paired_uncertainty(daily: pd.DataFrame) -> pd.DataFrame:
    """Pair arms within dates and resample blocks of whole trading days."""
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="contribution_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"], axis=0).all().all():
        raise ValueError("Arms differ in their daily evaluation coverage")
    rng = np.random.default_rng(SEED)
    results = []
    for arm, reference in (("E15-A", "E3"), ("E15-B", "E3"), ("E15-B", "E15-A")):
        deltas, simulations = [], []
        for fold in FOLDS:
            diff = (sums.loc[fold.name, arm] - sums.loc[fold.name, reference]).to_numpy()
            n_rows = counts.loc[fold.name, "E3"].to_numpy()
            n_days = len(diff)
            starts = rng.integers(0, n_days, size=(DRAWS, (n_days + BLOCK_DAYS - 1) // BLOCK_DAYS))
            sample = ((starts[..., None] + np.arange(BLOCK_DAYS)) % n_days).reshape(DRAWS, -1)
            sample = sample[:, :n_days]
            draws = diff[sample].sum(axis=1) / n_rows[sample].sum(axis=1)
            delta = float(diff.sum() / n_rows.sum())
            deltas.append(delta)
            simulations.append(draws)
            results.append(interval(arm, reference, fold.name, delta, draws))
        results.append(interval(arm, reference, "equal_year", np.mean(deltas),
                                np.mean(simulations, axis=0)))
    return pd.DataFrame(results)


def interval(arm: str, reference: str, period: str, point: float, draws: np.ndarray) -> dict:
    low, high = np.quantile(draws, [.025, .975])
    return {"arm": arm, "reference": reference, "period": period, "delta": point,
            "ci95_low": low, "ci95_high": high}


def run(data_dir: Path, f0_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a fresh E15 output directory: {output}")
    target_ids, reference_ids = load_metadata(data_dir)
    end = FOLDS[-1].eval_end
    market = load_market(data_dir).loc[lambda frame: frame["date"].le(end)]
    f0 = load_f0_dataset(f0_path).loc[lambda frame: frame["date"].le(end)]
    if not evaluation_keys(market).equals(evaluation_keys(f0)):
        raise ValueError("Historical F0 must exactly cover the Top100 market snapshot keys")
    ranked, coverage = rank_panel(f0, reference_ids, target_ids)
    e1 = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3 = load_fixed_rule(DEFAULT_E3_SUMMARY, e1["config_id"])
    expected = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    saved = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, e1)
    metrics, predictions, missingness = [], [], []
    for fold in FOLDS:
        train, valid = split_market(market, target_ids, fold)
        if set(train["stock_id"]) != set(reference_ids):
            raise ValueError(f"Training universe differs in {fold.name}")
        train_features = join_f0(train, ranked, include_label=False)
        valid_features = join_f0(valid, ranked, include_label=False)
        for split, frame in (("train", train_features), ("validation", valid_features)):
            for field in (*F0_FEATURES, *RANK_NAMES.values()):
                missing = int(frame[field].isna().sum())
                missingness.append({"fold": fold.name, "split": split, "feature": field,
                                    "n": len(frame), "missing": missing,
                                    "missing_fraction": missing / len(frame)})
        original_p = align_saved_probabilities(valid, saved, fold.name, SEED)
        tai = valid_features["taiex_return"].to_numpy()
        ret = ((valid[CLOSE] - valid[OPEN]) / valid[OPEN]).to_numpy()
        actual_up = ret > 0
        if not np.array_equal(actual_up, valid["y"].to_numpy()):
            raise ValueError(f"Historical labels disagree with intraday direction in {fold.name}")
        for arm, (x_train, x_valid) in transformed_inputs(train_features, valid_features).items():
            if not x_train.columns.equals(x_valid.columns):
                raise ValueError("Training and validation encoded columns differ")
            model = xgb.train({**e1["params"], "seed": SEED},
                              xgb.DMatrix(x_train, label=train["y"].to_numpy()),
                              num_boost_round=e1["num_boost_round"])
            p_up = model.predict(xgb.DMatrix(x_valid)).astype(float)
            label = regime_predictions(p_up, tai, rule)
            frame = valid[KEYS].assign(predicted_y=label)
            official = evaluate_predictions(valid, frame)
            if arm == "E3":
                score = expected["fold_official_scores_by_seed"][str(SEED)][fold.name]
                if (not np.array_equal(label, regime_predictions(original_p, tai, rule))
                        or not np.isclose(official["official_score"], score, atol=1e-12, rtol=0)
                        or not np.allclose(p_up, original_p, atol=1e-7, rtol=0)):
                    raise ValueError(f"Retrained E3 did not reproduce the archived {fold.name} control")
            metrics.append({"fold": fold.name, "arm": arm, **official,
                            "predicted_up_rate": label.mean(), "brier": np.mean((p_up - actual_up)**2),
                            "p_mean": p_up.mean(), "p_std": p_up.std(),
                            "p10": np.quantile(p_up, .1), "p50": np.median(p_up),
                            "p90": np.quantile(p_up, .9), "encoded_features": x_train.shape[1]})
            predictions.append(frame.assign(fold=fold.name, arm=arm, p_up=p_up,
                                            taiex_return=tai, actual_up=actual_up.astype(int),
                                            realized_return=ret, contribution=(2 * label - 1) * ret))
            print(f"{fold.name} {arm}: {official['official_score']:+.8f}", flush=True)

    paired = pd.concat(predictions, ignore_index=True)
    daily = daily_metrics(paired)
    distribution = daily.groupby(["fold", "arm"]).agg(
        trading_days=("date", "size"), positive_day_rate=("daily_score", lambda x: x.gt(0).mean()),
        daily_std=("daily_score", "std"), daily_p10=("daily_score", lambda x: x.quantile(.1)),
        daily_median=("daily_score", "median"), daily_p90=("daily_score", lambda x: x.quantile(.9)),
        mean_spearman=("spearman", "mean"), median_spearman=("spearman", "median"),
        defined_spearman_days=("spearman", "count"))
    fold_metrics = pd.DataFrame(metrics).join(distribution, on=["fold", "arm"])
    control = fold_metrics.loc[fold_metrics["arm"].eq("E3")].set_index("fold")["official_score"]
    fold_metrics["delta_e3"] = fold_metrics["official_score"] - fold_metrics["fold"].map(control)
    means = {arm: cv_score(dict(zip(group["fold"], group["official_score"])))
             for arm, group in fold_metrics.groupby("arm")}
    inputs = (f0_path, DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY,
              data_dir / "info.json", Path(__file__), ROOT / "shared/preprocessing.py",
              ROOT / "shared/evaluation.py", ROOT / "shared/features.py", ROOT / "shared/splits.py")
    summary = {"seed": SEED, "params": e1["params"], "rounds": e1["num_boost_round"],
               "taiex_rule": rule, "equal_year_scores": means,
               "equal_year_delta_e3": {arm: score - means["E3"] for arm, score in means.items()},
               "reference_ids": reference_ids, "stock_fields": STOCK_FIELDS,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "xgboost": xgb.__version__},
               "input_sha256": {path.name if path not in (DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY)
                                else f"{path.parent.name}/{path.name}": _sha256(path) for path in inputs},
               "market_sha256": {stock_id: _sha256(data_dir / "market" / f"{stock_id}.csv")
                                  for stock_id in reference_ids}}
    uncertainty = paired_uncertainty(daily)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("fold_metrics", fold_metrics), ("validation_predictions", paired),
                        ("daily_metrics", daily), ("paired_uncertainty", uncertainty),
                        ("reference_coverage", coverage), ("missingness", pd.DataFrame(missingness))):
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
