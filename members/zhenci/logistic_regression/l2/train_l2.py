"""Train one return-weighted Logistic Regression per locked historical fold."""

import argparse
import sys
import warnings
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from members.zhenci.logistic_regression.l1 import analysis as l1_analysis
from members.zhenci.logistic_regression.l1.export_phase1 import (
    DEFAULT_SUBMISSION, PHASE1, PHASE1_B, SUBMISSION_B, UPDATED_F0,
)
from members.zhenci.logistic_regression.l1.model import PARAMS, digest, explain, read_json, write_json
from members.zhenci.logistic_regression.l1.train_l1 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY, E20, F0, OUTPUT as L1_OUTPUT,
    PREVIOUS, archived_controls, decisions, load_history, match_archive, references,
)
from members.zhenci.logistic_regression.l2 import diagnostics as diag
from members.zhenci.logistic_regression.l2.report import render_report
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e2 import train_e2_weighted
from shared.data_loader import KEYS
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, SEED, split_market

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
L1_SUMMARY = L1_OUTPUT / "summary.json"
E17_OUTPUT = ROOT / "members/zhenci/xgboost/e17/outputs"
PHASE1_REPORTED = {"L1-A": -0.000119, "L1-B": 0.000304, "E1": -0.00031791,
                   "E3": -0.00006194, "E17-B": 0.000374}


def checked_l1_summary(rule):
    summary = read_json(L1_SUMMARY)
    if (summary["model"] != "LogisticRegression" or summary["params"] != PARAMS or not summary["l2"]
            or summary["seed"] != SEED or summary["features"] != list(F0_FEATURES)
            or summary["folds"] != [fold.name for fold in FOLDS]
            or summary["thresholds"] != {"L1-A": .5, "L1-B": rule}
            or summary["versions"]["scikit_learn"] != sklearn.__version__):
        raise ValueError("L1's archived recipe differs from the controlled L2 recipe")
    for name in (F0.relative_to(ROOT).as_posix(), "shared/features.py", "shared/preprocessing.py",
                 "shared/splits.py", "shared/evaluation.py"):
        if digest(ROOT / name) != summary["input_sha256"][name]:
            raise ValueError(f"L1 F0 or shared protocol input changed: {name}")
    return summary


def fit_weighted(raw, labels, weights):
    start = perf_counter()
    pre = BaselinePreprocessor().fit(raw)
    features = pre.transform(raw)
    model = LogisticRegression(**PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels, sample_weight=weights)
    if model.classes_.tolist() != [0, 1] or not 0 < model.n_iter_[0] < PARAMS["max_iter"]:
        raise RuntimeError("L2 did not converge within its fixed iteration budget")
    return model, pre, features, perf_counter() - start


def read_archived_predictions(path):
    return pd.read_csv(path, dtype={"stock_id": str, "fold": str}, parse_dates=["date"])


def archived_l1_model(fold, summary, pre):
    stored = L1_OUTPUT / "models" / fold
    for name in ("model.joblib", "preprocessor.json"):
        path = stored / name
        if digest(path) != summary["fold_artifact_sha256"][f"models/{fold}/{name}"]:
            raise ValueError(f"Archived L1 artifact differs for {fold}: {name}")
    if read_json(stored / "preprocessor.json") != pre.to_state():
        raise ValueError(f"L2 differs from L1 preprocessing on {fold} training rows")
    model = joblib.load(stored / "model.joblib")
    if model.get_params() != LogisticRegression(**PARAMS).get_params() or model.classes_.tolist() != [0, 1]:
        raise ValueError(f"Unexpected L1 model configuration for {fold}")
    return model


def archived_arm(source, arm, fold, valid, returns):
    table = source.loc[source["arm"].eq(arm) & source["fold"].eq(fold),
                       [*KEYS, "p_up", "predicted_y", "actual_up", "realized_return"]]
    aligned = match_archive(valid, table, arm)
    np.testing.assert_array_equal(aligned["actual_up"], valid["y"])
    np.testing.assert_allclose(aligned["realized_return"], returns, rtol=0, atol=1e-14)
    return aligned["p_up"].to_numpy(), aligned["predicted_y"].to_numpy()


def coefficient_comparison(fold, model, l1_model, features):
    names = features.columns.tolist()
    if l1_model.feature_names_in_.tolist() != names or model.feature_names_in_.tolist() != names:
        raise ValueError("Coefficient comparisons require identical encoded feature order")
    frame = pd.DataFrame({
        "fold": fold, "feature": [*names, "(intercept)"],
        "family": [*(l1_analysis.coefficient_family(name) for name in names), "intercept"],
        "L1": [*l1_model.coef_[0], l1_model.intercept_[0]],
        "L2": [*model.coef_[0], model.intercept_[0]],
    })
    return frame.assign(delta=frame["L2"] - frame["L1"], abs_L1=frame["L1"].abs(),
                        abs_L2=frame["L2"].abs(), sign_changed=frame["L1"] * frame["L2"] < 0)


def phase1_l1_diagnostics(rule):
    """Read saved Phase 1 scores and decisions, plus eligible F0, without loading outcomes."""
    left, right = (read_json(path / "export_manifest.json") for path in (PHASE1, PHASE1_B))
    for manifest, path in ((left, DEFAULT_SUBMISSION), (right, SUBMISSION_B)):
        if digest(path) != manifest["submission_sha256"]:
            raise ValueError(f"L1 Phase 1 submission changed: {path.name}")
    if (left["artifact_manifest_sha256"] != right["artifact_manifest_sha256"]
            or left["inference_f0_sha256"] != right["inference_f0_sha256"]
            or left["inference_f0_sha256"] != digest(UPDATED_F0) or right["decision_rule"] != rule):
        raise ValueError("Archived L1 Phase 1 rules or feature inputs differ")
    a, b = (pd.read_csv(path / "phase1_predictions.csv", dtype={"stock_id": str}, parse_dates=["date"])
            for path in (PHASE1, PHASE1_B))
    pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
    np.testing.assert_allclose(a["p_up"], b["p_up"], atol=1e-14, rtol=0)
    raw = join_f0(a, load_f0_dataset(UPDATED_F0), include_label=False)
    p = a["p_up"].to_numpy()
    calls = decisions(p, raw, rule)
    np.testing.assert_array_equal(calls["L1-A"], a["predicted_y"])
    np.testing.assert_array_equal(calls["L1-B"], b["predicted_y"])
    return diag.decision_diagnostics("Phase 1", "L1", p, calls["L1-A"], calls["L1-B"],
                                     raw["taiex_return"].to_numpy())


def run(output: Path):
    if output.exists():
        raise FileExistsError(f"Choose a new L2 output directory: {output}")
    targets, universe, market, f0 = load_history()
    _, _, rule, e1, e20, _, previous = archived_controls()
    l1_summary = checked_l1_summary(rule)
    l1_scores = read_archived_predictions(L1_OUTPUT / "validation_predictions.csv")
    e17 = read_archived_predictions(E17_OUTPUT / "validation_predictions.csv")
    e17_metrics = pd.read_csv(E17_OUTPUT / "fold_metrics.csv", dtype={"fold": str})
    l1_metrics = pd.read_csv(L1_OUTPUT / "fold_metrics.csv", dtype={"fold": str})
    weight_rows, fold_rows, daily_rows, pred_rows, loss_rows, timing = [], [], [], [], [], []
    coefficients, contribution_rows, family_rows, rule_rows = [], [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l2-", dir=output.parent) as temporary:
        staging = Path(temporary) / "outputs"
        staging.mkdir()
        for fold in FOLDS:
            train, valid = split_market(market, targets, fold)
            if set(train["stock_id"]) != set(universe):
                raise ValueError("L2 requires all 100 training stocks")
            raw_train, raw_valid = (join_f0(rows, f0, include_label=False) for rows in (train, valid))
            y_train, y_valid = train["y"].to_numpy(), valid["y"].to_numpy()
            weights, normalizer = diag.return_weights(train)
            returns = diag.realized_returns(valid)
            valid_weights = np.abs(returns) / normalizer
            weight_rows.append(diag.describe_weights(fold.name, y_train, weights, normalizer))
            model, pre, x_train, fit_seconds = fit_weighted(raw_train, y_train, weights)
            l1_model = archived_l1_model(fold.name, l1_summary, pre)
            start = perf_counter()
            x_valid = pre.transform(raw_valid)
            p_valid = model.predict_proba(x_valid)[:, 1]
            inference_seconds = perf_counter() - start
            timing.append(dict(fold=fold.name, fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                               optimizer_iterations=int(model.n_iter_[0]), converged=True))
            folder = staging / "models" / fold.name
            folder.mkdir(parents=True)
            joblib.dump(model, folder / "model.joblib")
            write_json(folder / "preprocessor.json", pre.to_state())

            old_p = l1_model.predict_proba(x_valid)[:, 1]
            coefficients.append(coefficient_comparison(fold.name, model, l1_model, x_valid))
            for name, fitted, probability in (("L1", l1_model, old_p), ("L2", model, p_valid)):
                parts, logit = explain(fitted, x_valid, probability)
                features, families, _ = l1_analysis.contribution_tables(
                    fold.name, valid[KEYS], y_valid, probability, x_valid, parts, logit, fitted.intercept_[0])
                contribution_rows.append(features.assign(model=name))
                family_rows.append(families.assign(model=name))
                loss_rows.append(diag.loss_metrics(fold.name, name, "train", y_train,
                                                   fitted.predict_proba(x_train)[:, 1], weights))

            base = references(valid, raw_valid, fold.name, e1, e20, previous, rule)
            archive_metrics = l1_metrics.loc[l1_metrics["fold"].eq(fold.name)].set_index("arm")
            archive_e17 = e17_metrics.loc[(e17_metrics["fold"].eq(fold.name))
                                          & (e17_metrics["arm"].eq("E17-B"))].iloc[0]
            combined = {arm: base[arm] for arm in ("E1", "E3")}
            combined["E17-B"] = archived_arm(e17, "E17-B", fold.name, valid, returns)
            for arm, calls in decisions(old_p, raw_valid, rule).items():
                p, saved = archived_arm(l1_scores, arm, fold.name, valid, returns)
                np.testing.assert_allclose(p, old_p, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(saved, calls)
                combined[arm] = p, saved
            combined.update({name.replace("L1", "L2"): (p_valid, calls)
                             for name, calls in decisions(p_valid, raw_valid, rule).items()})
            this_daily = []
            for arm, (p, calls) in combined.items():
                metrics, predictions, days = l1_analysis.score_arm(fold.name, arm, valid, returns, p, calls)
                if arm in archive_metrics.index:
                    np.testing.assert_allclose(metrics["official_score"], archive_metrics.at[arm, "official_score"],
                                               rtol=0, atol=1e-14)
                if arm == "E17-B":
                    np.testing.assert_allclose(metrics["official_score"], archive_e17["official_score"],
                                               rtol=0, atol=1e-14)
                loss = diag.loss_metrics(fold.name, arm, "validation", y_valid, p, valid_weights)
                fold_rows.append({**metrics, "direction_auc": loss["direction_auc"]})
                loss_rows.append(loss)
                score_type = ("return_weighted" if arm.startswith("L2") else
                              "daily_class_weighted" if arm == "E17-B" else "direction_probability")
                pred_rows.append(predictions.rename(columns={"p_up": "score"}).assign(score_type=score_type))
                this_daily.append(days)
            days = pd.concat(this_daily, ignore_index=True)
            daily_rows.append(days)
            taiex = raw_valid["taiex_return"].to_numpy()
            regime = np.where(np.isnan(taiex), "missing", np.where(taiex < 0, "negative", "nonnegative"))
            regimes = valid[["date"]].assign(regime=regime).drop_duplicates()
            days = days.merge(regimes, on="date", validate="many_to_one")
            for name in ("L1", "L2"):
                p, a = combined[f"{name}-A"]
                _, b = combined[f"{name}-B"]
                rule_rows.append(diag.decision_diagnostics(fold.name, name, p, a, b, taiex, returns, days))
            print(f"{fold.name} converged in {model.n_iter_[0]} iterations", flush=True)

        metrics = pd.DataFrame(fold_rows)
        daily = pd.concat(daily_rows, ignore_index=True)
        daily["month"] = daily["date"].dt.strftime("%Y-%m")
        daily["block"] = daily["fold"] + "-B" + ((daily["date"].dt.month - 1) // 4 + 1).astype(str)
        monthly = l1_analysis.periods(daily, "month")
        blocks = l1_analysis.periods(daily, "block")
        intervals = diag.paired_intervals(daily)
        overall, selection, candidate = diag.summarize(metrics, blocks, intervals)
        # Outcomes provided by the user enter the post-selection diagnostic and report only.
        phase1 = phase1_l1_diagnostics(rule)
        phase1["reported_fixed_score"] = PHASE1_REPORTED["L1-A"]
        phase1["reported_taiex_score"] = PHASE1_REPORTED["L1-B"]
        phase1.loc[phase1["regime"].ne("all"), ["reported_fixed_score", "reported_taiex_score"]] = np.nan
        coef = pd.concat(coefficients, ignore_index=True)
        long_coef = coef.melt(id_vars=["fold", "feature", "family"], value_vars=["L1", "L2"],
                              var_name="model", value_name="coefficient")
        stability = pd.concat([l1_analysis.coefficient_stability(long_coef.loc[long_coef["model"].eq(name)])
                               .assign(model=name) for name in ("L1", "L2")], ignore_index=True)
        tables = {
            "fold_metrics": metrics, "equal_year_scores": overall, "phase1_screen": selection,
            "daily_scores": daily, "monthly_scores": monthly, "four_month_scores": blocks,
            "paired_uncertainty": intervals, "loss_and_score_distributions": pd.DataFrame(loss_rows),
            "runtime": pd.DataFrame(timing), "sample_weights": pd.DataFrame(weight_rows),
            "coefficients": coef, "coefficient_stability": stability,
            "feature_contributions": pd.concat(contribution_rows, ignore_index=True),
            "family_contributions": pd.concat(family_rows, ignore_index=True),
            "threshold_sensitivity": pd.concat(rule_rows, ignore_index=True),
            "phase1_l1_threshold_sensitivity": phase1,
            "phase1_reported_scores": pd.DataFrame(PHASE1_REPORTED.items(), columns=["arm", "reported_score"]),
            "validation_predictions": pd.concat(pred_rows, ignore_index=True),
        }
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv", index=False)
        source_paths = (F0, L1_SUMMARY, L1_OUTPUT / "validation_predictions.csv", L1_OUTPUT / "fold_metrics.csv",
                        DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY,
                        E20 / "validation_predictions.csv", PREVIOUS / "paired_rows.csv",
                        E17_OUTPUT / "validation_predictions.csv", E17_OUTPUT / "fold_metrics.csv",
                        Path(train_e2_weighted.__file__), *sorted(HERE.glob("*.py")))
        write_json(staging / "summary.json", {
            "candidate": candidate, "selection_policy": diag.SELECTION_POLICY, "seed": SEED, "params": PARAMS,
            "weighting": "abs(adjusted intraday return) divided by its fold-training mean",
            "preprocessing": "unweighted L1 preprocessing verified against the saved fold states",
            "thresholds": {"L2-A": .5, "L2-B": rule}, "features": list(F0_FEATURES),
            "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "bootstrap": {"draws": diag.DRAWS, "block_days": diag.BLOCK_DAYS},
            "input_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
            "fold_artifact_sha256": {path.relative_to(staging).as_posix(): digest(path)
                                     for path in sorted((staging / "models").glob("*/*"))},
        })
        (staging / "L2_report.md").write_text(render_report(tables, candidate), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"L2 report: {output / 'L2_report.md'}\nPhase 1 candidate: {candidate or 'none'}")
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
