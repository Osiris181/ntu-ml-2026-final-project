"""Train and evaluate L1 using the three locked historical folds."""

import argparse
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn

from members.zhenci.logistic_regression.l1 import analysis
from members.zhenci.logistic_regression.l1.model import PARAMS, digest, explain, fit_l1, read_json, write_json
from members.zhenci.logistic_regression.l1.report import report
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY, align_saved_probabilities,
    checked_source_score, load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import CLOSE, OPEN, cv_score
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES
from shared.splits import FOLDS, SEED, split_market

DATA = ROOT / "data"
F0 = DATA / "processed/f0.csv"
OUTPUT = Path(__file__).resolve().with_name("outputs")
XGBOOST = ROOT / "members/zhenci/xgboost"
E20 = XGBOOST / "e20/outputs"
E19 = XGBOOST / "e19/outputs"
PREVIOUS = XGBOOST / "diagnostics/previous_day_gap_outputs"
YEARS = [fold.name for fold in FOLDS]


def load_history():
    targets, universe = load_metadata(DATA)
    market = load_market(DATA)
    market = market.loc[market["date"].le(FOLDS[-1].eval_end), [*KEYS, OPEN, CLOSE, "y"]]
    prices = market[[OPEN, CLOSE]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Historical adjusted prices must be finite and positive")
    if not market["y"].eq(market[CLOSE].gt(market[OPEN])).all():
        raise ValueError("Historical labels must match adjusted close > adjusted open")
    return targets, universe, market, load_f0_dataset(F0)


def decisions(p, raw, rule):
    return {"L1-A": (p > .5).astype(int),
            "L1-B": regime_predictions(p, raw["taiex_return"].to_numpy(), rule)}


def match_archive(valid, archived, key_description):
    validate_keys(archived)
    aligned = valid[KEYS].merge(archived, on=KEYS, how="left", sort=False,
                                validate="one_to_one", indicator=True)
    if len(archived) != len(valid) or not aligned["_merge"].eq("both").all():
        raise ValueError(f"Archived {key_description} does not exactly match the L1 validation keys")
    return aligned


def archived_controls():
    e1_summary = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3_summary = load_fixed_rule(DEFAULT_E3_SUMMARY, e1_summary["config_id"])
    e3_scores = next(row for row in e3_summary["comparisons"] if row["config_id"] == rule["config_id"])
    e1 = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, e1_summary)
    e20 = pd.read_csv(E20 / "validation_predictions.csv", dtype={"stock_id": str, "fold": str},
                      parse_dates=["date"])
    e20_summary = read_json(E20 / "summary.json")
    if (e20_summary["decision_rule"] != rule or e20_summary["seed"] != SEED
            or e20_summary["primary_rounds"] != 100):
        raise ValueError("Unexpected archived E20 configuration")
    previous = pd.read_csv(PREVIOUS / "paired_rows.csv", dtype={"stock_id": str, "period": str},
                           parse_dates=["date", "source_date"])
    return (e1_summary, e3_scores, rule, e1, e20.loc[e20["rounds"].eq(100)], e20_summary,
            previous.loc[previous["period"].isin(YEARS)])


def references(valid, raw, fold, e1, e20, previous, rule):
    p = align_saved_probabilities(valid, e1, fold, SEED)
    control = e20.loc[e20["fold"].eq(fold)]
    e3 = match_archive(valid, control.loc[control["arm"].eq("E3"), KEYS + ["p_up", "predicted_y"]], "E3")
    e3_calls = regime_predictions(p, raw["taiex_return"].to_numpy(), rule)
    np.testing.assert_allclose(p, e3["p_up"], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(e3_calls, e3["predicted_y"])
    e20_rows = match_archive(valid, control.loc[control["arm"].eq("E20"), KEYS + ["p_up", "predicted_y"]], "E20")
    np.testing.assert_array_equal(e20_rows["predicted_y"],
                                  regime_predictions(e20_rows["p_up"], raw["taiex_return"], rule))
    previous_rows = match_archive(
        valid, previous.loc[previous["period"].eq(fold), KEYS + ["source_date", "candidate"]],
        "previous-direction candidate")
    if not previous_rows["source_date"].lt(previous_rows["date"]).all():
        raise ValueError("Invalid archived previous-session dates")
    np.testing.assert_array_equal(previous_rows["candidate"], raw["intraday_return_1d"].gt(0))
    return {"E1": (p, (p > .5).astype(int)), "E3": (p, e3_calls),
            "E20": (e20_rows["p_up"].to_numpy(), e20_rows["predicted_y"].to_numpy()),
            "Previous direction": (None, previous_rows["candidate"].to_numpy())}


def archive_probability_metrics():
    tables = []
    for directory in (E19, E20):
        table = pd.read_csv(directory / "checkpoint_metrics.csv", dtype={"fold": str})
        if directory == E19:
            table = table.loc[table["rounds"].eq(100)].assign(arm="E3")
        else:
            table = table.loc[table["rounds"].eq(100) & table["arm"].eq("E20")]
        tables.append(table.filter(regex=r"^(arm|fold|(train|train_top50|validation)_(auc|bce|brier))$"))
    return pd.concat(tables, ignore_index=True)


def run(output: Path):
    if output.exists():
        raise FileExistsError(f"Choose a new L1 output directory: {output}")
    targets, universe, market, f0 = load_history()
    e1_summary, e3_scores, rule, e1, e20, e20_summary, previous = archived_controls()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l1-", dir=output.parent) as temporary:
        staging = Path(temporary) / "outputs"
        staging.mkdir()
        scores, rows, daily, probabilities, timings = [], [], [], [], []
        coefficients, features, groups, examples, correlations = [], [], [], [], []
        for fold in FOLDS:
            train, valid = split_market(market, targets, fold)
            if set(train["stock_id"]) != set(universe):
                raise ValueError("Each L1 fold requires the full original training universe")
            raw_train = join_f0(train, f0, include_label=False)
            raw_valid = join_f0(valid, f0, include_label=False)
            train_y, valid_y = train["y"].to_numpy(), valid["y"].to_numpy()
            model, pre, x_train, fit_seconds = fit_l1(raw_train, train_y)
            started = perf_counter()
            x_valid = pre.transform(raw_valid)
            valid_p = model.predict_proba(x_valid)[:, 1]
            inference_seconds = perf_counter() - started
            train_p = model.predict_proba(x_train)[:, 1]
            timings.append(dict(fold=fold.name, fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                                optimizer_iterations=int(model.n_iter_[0]), converged=True))

            folder = staging / "models" / fold.name
            folder.mkdir(parents=True)
            joblib.dump(model, folder / "model.joblib")
            write_json(folder / "preprocessor.json", pre.to_state())
            coefficients.append(pd.DataFrame({
                "fold": fold.name, "feature": [*x_train.columns, "(intercept)"],
                "family": [*(analysis.coefficient_family(name) for name in x_train), "intercept"],
                "coefficient": [*model.coef_[0], model.intercept_[0]],
            }))
            correlations.append(analysis.correlation_pairs(x_train, fold.name))
            parts, logit = explain(model, x_valid, valid_p)
            feat, family, sample = analysis.contribution_tables(
                fold.name, valid[KEYS], valid_y, valid_p, x_valid, parts, logit, model.intercept_[0])
            features.append(feat)
            groups.append(family)
            examples.append(sample)

            historical = references(valid, raw_valid, fold.name, e1, e20, previous, rule)
            train_calls = decisions(train_p, raw_train, rule)
            valid_calls = decisions(valid_p, raw_valid, rule)
            top50 = train["stock_id"].isin(targets).to_numpy()
            constant = train_y.mean()
            for arm in analysis.L1_ARMS:
                for split, y, p, calls in (
                    ("train_top100", train_y, train_p, train_calls[arm]),
                    ("train_top50", train_y[top50], train_p[top50], train_calls[arm][top50]),
                ):
                    probabilities.append(dict(fold=fold.name, arm=arm, split=split,
                                              **analysis.probability_metrics(y, p, calls, constant)))
            returns = ((valid[CLOSE] - valid[OPEN]) / valid[OPEN]).to_numpy()
            combined = {**{arm: (valid_p, calls) for arm, calls in valid_calls.items()}, **historical}
            for arm, (p, calls) in combined.items():
                metric, frame, days = analysis.score_arm(fold.name, arm, valid, returns, p, calls)
                scores.append(metric)
                rows.append(frame)
                daily.append(days)
                if p is not None:
                    probabilities.append(dict(fold=fold.name, arm=arm, split="validation",
                                              **analysis.probability_metrics(valid_y, p, calls, constant)))
                if arm in ("E1", "E3"):
                    recorded = e1_summary if arm == "E1" else e3_scores
                    checked_source_score(recorded["fold_official_scores_by_seed"], SEED, fold.name,
                                         metric["official_score"], arm)
            print(f"{fold.name} converged in {model.n_iter_[0]} iterations", flush=True)

        metrics = pd.DataFrame(scores)
        e20_scores = metrics.loc[metrics["arm"].eq("E20")].set_index("fold")["official_score"].to_dict()
        np.testing.assert_allclose(cv_score(e20_scores), e20_summary["equal_year_scores"]["E20"],
                                   atol=1e-12, rtol=0)
        days = pd.concat(daily, ignore_index=True)
        days["month"] = days["date"].dt.strftime("%Y-%m")
        days["block"] = days["fold"] + "-B" + ((days["date"].dt.month - 1) // 4 + 1).astype(str)
        monthly = analysis.periods(days, "month")
        blocks = analysis.periods(days, "block")
        overall, selected = analysis.summarize(metrics, blocks)
        uncertainty = analysis.bootstrap(days)
        probs = pd.DataFrame(probabilities)
        fit = probs.loc[probs["arm"].isin(analysis.L1_ARMS)]
        gaps = fit.pivot(index=["fold", "arm"], columns="split", values=["bce", "auc", "brier", "accuracy"])
        gaps = pd.DataFrame({f"validation_minus_{split}_{metric}": gaps[metric]["validation"] - gaps[metric][split]
                             for split in ("train_top100", "train_top50")
                             for metric in ("bce", "auc", "brier", "accuracy")}).reset_index()
        coef = pd.concat(coefficients, ignore_index=True)
        coef["abs_coefficient"] = coef["coefficient"].abs()
        stability = analysis.coefficient_stability(coef)
        old_probs = archive_probability_metrics()
        by_family = pd.concat(groups, ignore_index=True)
        corr = pd.concat(correlations, ignore_index=True)
        timing = pd.DataFrame(timings)
        tables = {
            "fold_metrics": metrics, "equal_year_scores": overall, "probability_metrics": probs,
            "probability_metric_gaps": gaps, "archived_xgboost_probability_metrics": old_probs,
            "validation_predictions": pd.concat(rows, ignore_index=True), "runtime": timing,
            "daily_scores": days, "monthly_scores": monthly, "four_month_scores": blocks,
            "paired_uncertainty": uncertainty, "coefficients": coef, "coefficient_stability": stability,
            "numerical_correlations": corr, "feature_contributions": pd.concat(features, ignore_index=True),
            "family_contributions": by_family, "prediction_explanations": pd.concat(examples, ignore_index=True),
        }
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv", index=False)
        sources = [F0, DATA / "info.json", DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY,
                   E19 / "checkpoint_metrics.csv", E20 / "checkpoint_metrics.csv", E20 / "summary.json",
                   E20 / "validation_predictions.csv", PREVIOUS / "paired_rows.csv", ROOT / "shared/features.py",
                   ROOT / "shared/preprocessing.py", ROOT / "shared/evaluation.py", ROOT / "shared/splits.py",
                   *Path(__file__).parent.glob("*.py"), *(DATA / "market" / f"{stock}.csv" for stock in universe)]
        write_json(staging / "summary.json", {
            "model": "LogisticRegression", "params": PARAMS, "l2": True, "seed": SEED, "folds": YEARS,
            "features": list(F0_FEATURES), "thresholds": {"L1-A": .5, "L1-B": rule}, "candidate": selected,
            "selection_policy": analysis.SELECTION_POLICY, "bootstrap_draws": analysis.DRAWS,
            "block_trading_days": analysis.BLOCK_DAYS,
            "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
            "input_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(sources)},
            "fold_artifact_sha256": {path.relative_to(staging).as_posix(): digest(path)
                                     for path in sorted((staging / "models").glob("*/*"))},
        })
        (staging / "L1_report.md").write_text(
            report(metrics, overall, uncertainty, probs, gaps, old_probs, stability, by_family, corr, timing, selected),
            encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"L1 report: {output / 'L1_report.md'}\nPhase 1 candidate: {selected or 'none'}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
