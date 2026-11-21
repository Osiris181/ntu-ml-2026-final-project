"""Fit and compare the two predefined L3 decision rules on historical folds."""

import argparse
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
from sklearn.linear_model import Ridge

from members.zhenci.logistic_regression.l1.analysis import coefficient_stability, periods, score_arm
from members.zhenci.logistic_regression.l1.model import digest, read_json, write_json
from members.zhenci.logistic_regression.l1.train_l1 import (
    DATA, DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY, F0, OUTPUT as L1, load_history,
)
from members.zhenci.logistic_regression.l2.diagnostics import realized_returns
from members.zhenci.logistic_regression.l2.train_l2 import OUTPUT as L2, read_archived_predictions
from members.zhenci.ridge_regression.l3 import diagnostics as diag
from members.zhenci.ridge_regression.l3.report import render_report
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e4.train_e4 import (
    align_saved_probabilities, load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from shared.data_loader import KEYS
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, SEED, split_market

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
PARAMS = {"alpha": 1.0, "fit_intercept": True, "solver": "auto"}
DELTA_MULTIPLIER = .05
PHASE_REPLAY = ROOT / "members/zhenci/xgboost/diagnostics/e1_e3_gap_outputs"


def fit_ridge(raw, returns):
    start = perf_counter()
    pre = BaselinePreprocessor().fit(raw)
    features = pre.transform(raw)
    model = Ridge(**PARAMS).fit(features, returns)
    return model, pre, perf_counter() - start


def taiex_thresholds(taiex, delta):
    if np.isinf(taiex).any() or not np.isfinite(delta) or delta < 0:
        raise ValueError("TAIEX values and the training-derived threshold must be valid")
    return np.where(np.isnan(taiex), 0., np.where(taiex < 0, delta, -delta))


def frozen_reference(directory, fold, summary, pre):
    folder = directory / "models" / fold
    for name in ("model.joblib", "preprocessor.json"):
        if digest(folder / name) != summary["fold_artifact_sha256"][f"models/{fold}/{name}"]:
            raise ValueError(f"Archived model checksum differs: {folder / name}")
    if read_json(folder / "preprocessor.json") != pre.to_state():
        raise ValueError(f"Training-only preprocessing differs from the archived fold: {folder}")
    return joblib.load(folder / "model.joblib")


def control_rows(source, arm, fold, valid, actual, score_column):
    subset = source.loc[source["arm"].eq(arm) & source["fold"].eq(fold)]
    # Index equality checks coverage, including duplicates and unexpected keys.
    index = pd.MultiIndex.from_frame(valid[KEYS])
    subset = subset.set_index(KEYS, verify_integrity=True)
    if len(subset) != len(index) or not index.isin(subset.index).all():
        raise ValueError(f"{arm} archive differs from L3's validation keys for {fold}")
    aligned = subset.reindex(index)
    np.testing.assert_array_equal(aligned["actual_up"], valid["y"])
    np.testing.assert_allclose(aligned["realized_return"], actual, rtol=0, atol=1e-14)
    return aligned[score_column].to_numpy(), aligned["predicted_y"].to_numpy()


def run(output: Path):
    if output.exists():
        raise FileExistsError(f"Choose a new L3 output directory: {output}")
    targets, universe, market, f0 = load_history()
    old1, old2 = (read_json(path / "summary.json") for path in (L1, L2))
    for old in (old1, old2):
        if (old["input_sha256"][F0.relative_to(ROOT).as_posix()] != digest(F0)
                or old["features"] != list(F0_FEATURES)
                or old["versions"]["scikit_learn"] != sklearn.__version__):
            raise ValueError("Archived linear references differ in F0 or runtime")
    e1_historical = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3_historical = load_fixed_rule(DEFAULT_E3_SUMMARY, e1_historical["config_id"])
    e3_selected = next(row for row in e3_historical["comparisons"] if row["config_id"] == rule["config_id"])
    e1_predictions = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, e1_historical)
    l1_pred, l2_pred = (read_archived_predictions(path / "validation_predictions.csv") for path in (L1, L2))
    l2_metrics = pd.read_csv(L2 / "fold_metrics.csv", dtype={"fold": str}).set_index(["fold", "arm"])
    score_rows, predictions, days, regressions, magnitude, residual = [], [], [], [], [], []
    coefficients, family_rows, regime_rows, runtimes = [], [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l3-", dir=output.parent) as temporary:
        staged = Path(temporary) / "outputs"
        staged.mkdir()
        for fold in FOLDS:
            train, valid = split_market(market, targets, fold)
            if set(train["stock_id"]) != set(universe):
                raise ValueError("L3 requires the original 100-stock training universe")
            raw_train, raw_valid = (join_f0(rows, f0, include_label=False) for rows in (train, valid))
            r_train, r_valid = (realized_returns(rows) for rows in (train, valid))
            train_mean = r_train.mean()
            mean_magnitude = np.abs(r_train).mean()
            delta = DELTA_MULTIPLIER * mean_magnitude
            model, pre, fit_seconds = fit_ridge(raw_train, r_train)
            start = perf_counter()
            x_valid = pre.transform(raw_valid)
            p_valid = model.predict(x_valid)
            inference_seconds = perf_counter() - start
            p_train = model.predict(pre.transform(raw_train))
            if not np.isfinite(p_train).all() or not np.isfinite(p_valid).all():
                raise ValueError("Ridge predicted nonfinite returns")
            fixed = (p_valid > 0).astype(int)
            taiex = raw_valid["taiex_return"].to_numpy()
            dynamic = (p_valid > taiex_thresholds(taiex, delta)).astype(int)
            subset = train["stock_id"].isin(targets).to_numpy()
            for name, p, actual in (("train_top100", p_train, r_train),
                                    ("train_top50", p_train[subset], r_train[subset]),
                                    ("validation", p_valid, r_valid)):
                regressions.append(diag.regression_metrics(fold.name, name, p, actual, train_mean))
            magnitude.append(diag.magnitude_groups(fold.name, p_valid, r_valid, fixed))
            residual.append(diag.residuals(fold.name, p_valid, r_valid))
            l1_model, l2_model = (frozen_reference(path, fold.name, archive, pre)
                                  for path, archive in ((L1, old1), (L2, old2)))
            coeff, family = diag.coefficient_tables(fold.name, model, l1_model, l2_model, x_valid, p_valid)
            coefficients.append(coeff)
            family_rows.append(family)
            folder = staged / "models" / fold.name
            folder.mkdir(parents=True)
            joblib.dump(model, folder / "model.joblib")
            write_json(folder / "preprocessor.json", pre.to_state())
            runtimes.append(dict(fold=fold.name, solver=getattr(model, "solver_", "unavailable"),
                                 fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                                 mean_abs_training_return=mean_magnitude, delta=delta, training_rows=len(train)))
            arms = {"L3-A": (p_valid, fixed), "L3-B": (p_valid, dynamic)}
            for name in diag.CONTROLS:
                archive, column = (l1_pred, "p_up") if name.startswith("L1") else (l2_pred, "score")
                arms[name] = control_rows(archive, name, fold.name, valid, r_valid, column)
            p_e1 = align_saved_probabilities(valid, e1_predictions, fold.name, SEED)
            for name in ("E1", "E3"):
                np.testing.assert_allclose(arms[name][0], p_e1, rtol=0, atol=1e-7)
                archive_p, archive_calls = control_rows(l1_pred, name, fold.name, valid, r_valid, "p_up")
                np.testing.assert_allclose(archive_p, arms[name][0], rtol=0, atol=1e-7)
                np.testing.assert_array_equal(archive_calls, arms[name][1])
            for name in ("L1-A", "L1-B"):
                cross_p, cross_calls = control_rows(l2_pred, name, fold.name, valid, r_valid, "score")
                np.testing.assert_allclose(cross_p, arms[name][0], rtol=0, atol=1e-14)
                np.testing.assert_array_equal(cross_calls, arms[name][1])
            fold_days = []
            for name, (score, calls) in arms.items():
                metrics, frame, daily = score_arm(fold.name, name, valid, r_valid, score, calls)
                if name in diag.CONTROLS:
                    np.testing.assert_allclose(metrics["official_score"], l2_metrics.at[(fold.name, name), "official_score"],
                                               rtol=0, atol=1e-14)
                if name in ("E1", "E3"):
                    archived = e1_historical if name == "E1" else e3_selected
                    expected = archived["fold_official_scores_by_seed"][str(SEED)][fold.name]
                    np.testing.assert_allclose(metrics["official_score"], expected, rtol=0, atol=1e-14)
                score_rows.append({**metrics, "direction_auc": diag.score_auc(valid["y"], score)})
                score_type = ("predicted_return" if name in diag.ARMS
                              else "return_weighted" if name.startswith("L2") else "direction_probability")
                predictions.append(frame.rename(columns={"p_up": "continuous_score"}).assign(score_type=score_type))
                fold_days.append(daily)
            daily = pd.concat(fold_days, ignore_index=True)
            days.append(daily)
            regime = np.where(np.isnan(taiex), "missing", np.where(taiex < 0, "negative", "nonnegative"))
            regimes = valid[["date"]].assign(regime=regime).drop_duplicates()
            regime_rows.append(diag.threshold_effect(fold.name, p_valid, r_valid, fixed, dynamic, taiex,
                daily.loc[daily["arm"].isin(diag.ARMS)].merge(regimes, on="date", validate="many_to_one"),
                mean_magnitude, delta))
            print(f"{fold.name} fitted Ridge with solver {runtimes[-1]['solver']}", flush=True)

        metrics, daily, reg = pd.DataFrame(score_rows), pd.concat(days, ignore_index=True), pd.DataFrame(regressions)
        daily["month"] = daily["date"].dt.strftime("%Y-%m")
        daily["block"] = daily["fold"] + "-B" + ((daily["date"].dt.month - 1) // 4 + 1).astype(str)
        blocks = periods(daily, "block")
        intervals = diag.paired_intervals(daily)
        overall, screen, candidate = diag.summarize(metrics, blocks, intervals)
        coef = pd.concat(coefficients, ignore_index=True)
        stability = coefficient_stability(coef.rename(columns={"L3_return_coef": "coefficient"}))
        gaps = reg.pivot(index="fold", columns="split", values=["mse", "rmse", "mae", "pearson", "spearman"])
        gaps = pd.DataFrame({f"validation_minus_{split}_{m}": gaps[m]["validation"] - gaps[m][split]
                             for split in ("train_top100", "train_top50")
                             for m in ("mse", "rmse", "mae", "pearson", "spearman")}).reset_index()
        inputs = (F0, L1 / "summary.json", L2 / "summary.json", L1 / "validation_predictions.csv",
                  L2 / "validation_predictions.csv", L2 / "fold_metrics.csv", DEFAULT_E1_SUMMARY,
                  DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY, ROOT / "shared/features.py",
                  ROOT / "shared/preprocessing.py", ROOT / "shared/splits.py", ROOT / "shared/evaluation.py",
                  DATA / "info.json", *(DATA / "market" / f"{stock}.csv" for stock in universe),
                  HERE / "train_l3.py", HERE / "diagnostics.py", HERE / "report.py",
                  *(directory / path for directory, old in ((L1, old1), (L2, old2)) for path in old["fold_artifact_sha256"]))
        write_json(staged / "summary.json", {
            "candidate": candidate, "selection_policy": diag.SELECTION_POLICY, "reference": diag.REFERENCE,
            "model": "sklearn.linear_model.Ridge", "params": PARAMS, "sample_weighting": "uniform",
            "target": "adjusted intraday return in decimal units",
            "seed": SEED, "features": list(F0_FEATURES), "delta_multiplier": DELTA_MULTIPLIER,
            "thresholds": {"L3-A": 0, "L3-B": "negative +delta, nonnegative -delta, missing 0"},
            "bootstrap": {"draws": diag.DRAWS, "block_days": diag.BLOCK_DAYS},
            "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "input_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
            "fold_artifact_sha256": {path.relative_to(staged).as_posix(): digest(path)
                                     for path in sorted((staged / "models").glob("*/*"))},
        })
        from members.zhenci.ridge_regression.l3.phase1_audit import audit_prior_submissions
        phase1, phase_sources = audit_prior_submissions(overall)
        write_json(staged / "phase1_reference_provenance.json", phase_sources)
        tables = {
            "fold_metrics": metrics, "equal_year_scores": overall, "daily_scores": daily,
            "monthly_scores": periods(daily, "month"), "four_month_scores": blocks,
            "validation_predictions": pd.concat(predictions, ignore_index=True), "phase1_screen": screen,
            "paired_uncertainty": intervals, "regression_metrics": reg, "regression_gaps": gaps,
            "prediction_magnitude_quintiles": pd.concat(magnitude, ignore_index=True),
            "residual_diagnostics": pd.DataFrame(residual), "coefficients": coef, "coefficient_stability": stability,
            "feature_family_contributions": pd.concat(family_rows, ignore_index=True),
            "threshold_sensitivity": pd.concat(regime_rows, ignore_index=True), "runtime": pd.DataFrame(runtimes),
            "phase1_reference_audit": phase1,
        }
        for name, table in tables.items():
            table.to_csv(staged / f"{name}.csv", index=False)
        (staged / "L3_report.md").write_text(render_report(tables, candidate), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    print(f"L3 report: {output / 'L3_report.md'}\nPhase 1 candidate: {candidate or 'none'}")
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
