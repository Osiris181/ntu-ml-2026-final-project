"""Fit the single fixed E20 model in each fold and compare it with frozen E3."""

from __future__ import annotations

import argparse
import json
import platform
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
from members.zhenci.xgboost.e19.diagnose_e19 import FOLD_SOURCE, daily_stats, distribution, inspect_trees, load_json, periodic
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import CLOSE, OPEN, cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, SEED, split_market

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
DATA = ROOT / "data"
F0 = DATA / "processed/f0.csv"
CHECKPOINTS = (10, 25, 50, 75, 100)
CHANGES = {"max_depth": 4, "min_child_weight": 10, "lambda": 5.0}


def paired_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    totals = daily.pivot(index=["fold", "rounds", "date"], columns="arm", values="score_sum")
    counts = daily.pivot(index=["fold", "rounds", "date"], columns="arm", values="n")
    if totals.isna().any().any() or not counts["E3"].equals(counts["E20"]):
        raise ValueError("Bootstrap requires paired stock-date coverage")
    rng = np.random.default_rng(SEED)
    rows = []
    for rounds in CHECKPOINTS:
        points, samples = [], []
        for fold in FOLDS:
            paired = totals.loc[(fold.name, rounds)]
            delta = (paired["E20"] - paired["E3"]).to_numpy()
            n = counts.loc[(fold.name, rounds), "E3"].to_numpy()
            days = len(n)
            starts = rng.integers(0, days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
            indices = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days).reshape(DRAWS, -1)[:, :days]
            points.append(float(delta.sum() / n.sum()))
            samples.append(delta[indices].sum(axis=1) / n[indices].sum(axis=1))
            rows.append({"rounds": rounds, **interval("E20", "E3", fold.name, points[-1], samples[-1])})
        rows.append({"rounds": rounds, **interval("E20", "E3", "equal_year", np.mean(points), np.mean(samples, axis=0))})
    return pd.DataFrame(rows)


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    challenger = {**recipe, "params": {**recipe["params"], **CHANGES}}
    if recipe["num_boost_round"] != CHECKPOINTS[-1]:
        raise ValueError("Unexpected archived round count")
    rule, e3 = load_fixed_rule(DEFAULT_E3_SUMMARY, recipe["config_id"])
    control = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    archived = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, recipe)
    source_path = FOLD_SOURCE / "summary.json"
    source = load_json(source_path)
    target_ids, all_ids = load_metadata(DATA)
    if (source["params"] != recipe["params"] or source["seed"] != SEED or source["arm_extras"]["E3"]
            or source["input_sha256"].get("data/processed/f0.csv") != _sha256(F0)
            or any(source["market_sha256"].get(stock) != _sha256(DATA / "market" / f"{stock}.csv") for stock in all_ids)):
        raise ValueError("Frozen E3 fold training inputs or recipe differ")
    market = load_market(DATA).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    f0 = load_f0_dataset(F0).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    metrics, days, predictions, trees, leaves, runtime = [], [], [], [], [], []
    files = [F0, DATA / "info.json", source_path, DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY,
             Path(__file__), ROOT / "shared/preprocessing.py", ROOT / "shared/evaluation.py"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="e20-", dir=output.parent) as temporary:
        out = Path(temporary) / "outputs"
        out.mkdir()
        for fold in FOLDS:
            train, valid = split_market(market, target_ids, fold)
            if set(train["stock_id"]) != set(all_ids):
                raise ValueError("Expected all 100 stocks in each training fold")
            frozen = FOLD_SOURCE / "models" / fold.name / "E3"
            state_path, model_path = frozen / "preprocessing.json", frozen / "model.json"
            files.extend([state_path, model_path])
            state = load_json(state_path)
            if state["extras"]:
                raise ValueError("E3 control must contain F0 only")
            pre = BaselinePreprocessor.from_state(state["base"])
            raw_train, raw_valid = (join_f0(rows, f0, include_label=False) for rows in (train, valid))
            history = raw_train[list(F0_FEATURES)]
            if BaselinePreprocessor().fit(history).to_state() != state["base"]:
                raise ValueError("Frozen preprocessor differs from fit on the current training keys")
            x_train, x_valid = pre.transform(history), pre.transform(raw_valid[list(F0_FEATURES)])
            dtrain, dvalid = (xgb.DMatrix(frame, nthread=4) for frame in (x_train, x_valid))
            e3_model = xgb.Booster(model_file=str(model_path))
            start = perf_counter()
            fitted = fit_classifier(challenger, x_train, train, pd.Series(1.0, index=train.index))
            runtime.append({"fold": fold.name, "train_rows": len(train), "fit_seconds": perf_counter() - start})
            e20_model = fitted.get_booster()
            folder = out / "models" / fold.name
            folder.mkdir(parents=True)
            e20_model.save_model(str(folder / "model.json"))
            (folder / "preprocessor.json").write_text(json.dumps(state["base"], indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")

            p_archive = align_saved_probabilities(valid, archived, fold.name, SEED)
            train_tai, valid_tai = (frame["taiex_return"].to_numpy() for frame in (raw_train, raw_valid))
            train_y, valid_y = train["y"].to_numpy(), valid["y"].to_numpy()
            returns = ((valid[CLOSE] - valid[OPEN]) / valid[OPEN]).to_numpy()
            if not np.array_equal(returns > 0, valid_y):
                raise ValueError("Validation labels differ from adjusted return direction")
            for arm, model in (("E3", e3_model), ("E20", e20_model)):
                model.set_param({"nthread": 4})
                if model.num_boosted_rounds() != CHECKPOINTS[-1]:
                    raise ValueError(f"{arm} did not use the fixed round budget")
                for rounds in CHECKPOINTS:
                    p_train = model.predict(dtrain, iteration_range=(0, rounds))
                    p_valid = model.predict(dvalid, iteration_range=(0, rounds))
                    train_calls = regime_predictions(p_train, train_tai, rule)
                    valid_calls = regime_predictions(p_valid, valid_tai, rule)
                    frame = valid[KEYS].assign(p_up=p_valid, predicted_y=valid_calls, actual_up=valid_y,
                                              realized_return=returns, contribution=(2 * valid_calls - 1) * returns)
                    official = evaluate_predictions(valid, frame[KEYS + ["predicted_y"]])["official_score"]
                    error = np.nan
                    if arm == "E3" and rounds == 100:
                        error = np.max(np.abs(p_valid - p_archive))
                        if (error > 1e-7 or not np.array_equal(valid_calls, regime_predictions(p_archive, valid_tai, rule))
                                or not np.isclose(official, control["fold_official_scores_by_seed"][str(SEED)][fold.name],
                                                  atol=1e-12, rtol=0)):
                            raise ValueError(f"Archived E3 mismatch for {fold.name}")
                    daily = daily_stats(frame).assign(arm=arm, fold=fold.name, rounds=rounds)
                    days.append(daily)
                    predictions.append(frame.assign(arm=arm, fold=fold.name, rounds=rounds))
                    metrics.append({"arm": arm, "fold": fold.name, "rounds": rounds, "train_n": len(train),
                        "validation_n": len(valid), "official_score": official, "archived_probability_error": error,
                        "mean_daily_spearman": daily["spearman"].mean(), "daily_score_std": daily["daily_score"].std(),
                        "negative_day_rate": daily["daily_score"].lt(0).mean(),
                        **distribution("train", train_y, p_train, train_calls),
                        **distribution("validation", valid_y, p_valid, valid_calls)})
                tree, leaf = inspect_trees(model, dtrain, dvalid, arm)
                if arm == "E20" and (tree["max_depth"].gt(CHANGES["max_depth"]).any()
                                      or leaf["hessian_cover"].lt(CHANGES["min_child_weight"] - 1e-5).any()):
                    raise ValueError("E20 tree complexity is inconsistent with the fixed configuration")
                trees.append(tree.assign(fold=fold.name))
                leaves.append(leaf.assign(fold=fold.name))
                print(f"{fold.name} {arm} at 100 rounds {official:+.8f}", flush=True)

        frame, daily = pd.DataFrame(metrics), pd.concat(days, ignore_index=True)
        frame["bce_gap"] = frame["validation_bce"] - frame["train_bce"]
        frame["auc_gap"] = frame["train_auc"] - frame["validation_auc"]
        frame["accuracy_gap"] = frame["train_accuracy"] - frame["validation_accuracy"]
        daily["month"] = daily["date"].dt.strftime("%Y-%m")
        daily["block"] = daily["fold"] + "-B" + ((daily["date"].dt.month - 1) // 4 + 1).astype(str)
        months, blocks = (pd.concat([periodic(group, key).assign(arm=arm) for arm, group in daily.groupby("arm")],
                                    ignore_index=True) for key in ("month", "block"))
        primary = frame.loc[frame["rounds"].eq(100)]
        scores = {arm: cv_score(dict(zip(group["fold"], group["official_score"]))) for arm, group in primary.groupby("arm")}
        summary = {"checkpoints": CHECKPOINTS, "primary_rounds": 100, "seed": SEED, "decision_rule": rule,
                   "E3_params": recipe["params"], "E20_params": challenger["params"], "equal_year_scores": scores,
                   "bootstrap_draws": DRAWS, "block_days": BLOCK_DAYS,
                   "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                                "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
                   "input_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in files}}
        datasets = {"checkpoint_metrics": frame, "daily_scores": daily, "monthly_scores": months, "four_month_scores": blocks,
                    "validation_predictions": pd.concat(predictions, ignore_index=True), "trees": pd.concat(trees, ignore_index=True),
                    "leaves": pd.concat(leaves, ignore_index=True), "paired_uncertainty": paired_intervals(daily),
                    "runtime": pd.DataFrame(runtime)}
        for name, table in datasets.items():
            table.to_csv(out / f"{name}.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        out.rename(output)
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
