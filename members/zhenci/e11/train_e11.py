"""E11: fixed XGBoost conditional-payoff experiment on the original F0 folds.

Run from final_project: uv run python -m members.zhenci.e11.train_e11
Only 2020-2025 original market/F0 data and saved E1 out-of-sample probabilities
are read. No deployment model is saved and no later-period labels are accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.train_e1 import join_f0
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import E1_SEEDS, FOLDS, SEED, split_market

RESULT = ROOT / "results/zhenci_xgboost_e11_expected_return_v1"
OPEN, CLOSE = "etl:adj_open", "etl:adj_close"
ARMS = ("e1", "e3", "previous_direction_candidate", "e11_direction_only",
        "e11_constant_payoffs", "e11_conditional_payoffs")
CAP_QUANTILE = 0.995
N_BOOTSTRAP = 3000
BLOCK_DAYS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_sha256(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((data_dir/"market").glob("*.csv")):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def training_returns(rows: pd.DataFrame) -> np.ndarray:
    prices = rows[[OPEN, CLOSE]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices[:, 0] <= 0).any():
        raise ValueError("Training prices must be finite and opens positive")
    realized = (prices[:, 1] - prices[:, 0]) / prices[:, 0]
    if not np.array_equal((realized > 0).astype(int), rows["y"].to_numpy(dtype=int)):
        raise ValueError("Training labels disagree with adjusted intraday direction")
    return realized


def expected_return(p_up: np.ndarray, q_flat_given_nonup: np.ndarray,
                    gain_if_up: np.ndarray, loss_if_down: np.ndarray) -> np.ndarray:
    """E[r|x] = P(up)*E[r|up] - P(down)*E[-r|down]. Flat payoff is zero."""
    arrays = [np.asarray(v, dtype=float) for v in
              (p_up, q_flat_given_nonup, gain_if_up, loss_if_down)]
    if any(v.ndim != 1 or v.shape != arrays[0].shape or not np.isfinite(v).all()
           for v in arrays):
        raise ValueError("Expected-return components must be aligned finite vectors")
    p, q, gain, loss = arrays
    if ((p < 0).any() or (p > 1).any() or (q < 0).any() or (q > 1).any()
            or (gain < 0).any() or (loss < 0).any()):
        raise ValueError("Invalid probability or conditional magnitude")
    return p*gain - (1-p)*(1-q)*loss


def _load_controls(f0_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    e1_dir = ROOT / "results/zheng_xgboost_e1_five_seeds_v1"
    with (e1_dir/"summary.json").open(encoding="utf-8") as file:
        e1 = json.load(file)
    with (ROOT/"results/zhenci_xgboost_e3_market_regime_v2/summary.json").open(
        encoding="utf-8") as file:
        e3 = json.load(file)
    if (e1["feature_order"] != list(F0_FEATURES)
            or e1["config_id"] != "xgboost_initial_v1"
            or e1["seeds"] != list(E1_SEEDS)
            or e1["num_boost_round"] != 100
            or e1["params"]["subsample"] != 1.0
            or e1["params"]["colsample_bytree"] != 1.0):
        raise ValueError("E1 control no longer has the original deterministic F0 configuration")
    rule = selected_rule(e3, e1["config_id"])
    saved = pd.read_csv(e1_dir/"validation_predictions.csv",
                        dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    validate_keys(saved.loc[saved["seed"].eq(SEED), KEYS])
    if (set(saved["seed"]) != set(E1_SEEDS)
            or set(saved["fold"]) != {fold.name for fold in FOLDS}
            or set(saved["config_id"]) != {e1["config_id"]}
            or saved.duplicated(["fold", "seed", *KEYS]).any()
            or saved.groupby(["fold", *KEYS])["seed"].nunique().ne(5).any()
            or saved.groupby(["fold", *KEYS])["p_up"].nunique().ne(1).any()):
        raise ValueError("Incomplete or seed-dependent E1 saved CV probabilities")
    saved = saved.loc[saved["seed"].eq(SEED)].copy()
    if not saved["predicted_y"].eq(saved["p_up"].gt(.5).astype(int)).all():
        raise ValueError("Saved E1 labels disagree with probabilities")
    return saved, e1, rule


def _fit_component(x_train: pd.DataFrame, labels: np.ndarray, mask: np.ndarray,
                   x_valid: pd.DataFrame, params: dict) -> np.ndarray:
    if int(mask.sum()) < 100 or len(np.unique(labels[mask])) < 2:
        # Regression labels can be continuous with distinct values; all components
        # require enough nonconstant conditional observations in each fold.
        raise ValueError("Insufficient conditional component training examples")
    dtrain = xgb.DMatrix(x_train.loc[mask], label=labels[mask])
    booster = xgb.train(params, dtrain, num_boost_round=100)
    predicted = booster.predict(xgb.DMatrix(x_valid))
    if predicted.shape != (len(x_valid),) or not np.isfinite(predicted).all():
        raise ValueError("Invalid conditional component output")
    return predicted


def _fold(fold, market: pd.DataFrame, f0: pd.DataFrame, target_ids: list[str],
          saved: pd.DataFrame, e1: dict, rule: dict) -> tuple[pd.DataFrame, dict]:
    train_rows, valid_rows = split_market(market, target_ids, fold)
    train = join_f0(train_rows, f0, include_label=True)
    valid = join_f0(valid_rows, f0, include_label=False)
    preprocessor = BaselinePreprocessor()
    x_train = preprocessor.fit_transform(train[list(F0_FEATURES)])
    x_valid = preprocessor.transform(valid[list(F0_FEATURES)])
    if x_train.columns.tolist() != x_valid.columns.tolist():
        raise ValueError("Train/validation F0 preprocessing differs")
    returns = training_returns(train_rows)
    up, down, flat = returns > 0, returns < 0, returns == 0
    gain = returns[up]
    loss = -returns[down]
    gain_cap, loss_cap = np.quantile(gain, CAP_QUANTILE), np.quantile(loss, CAP_QUANTILE)
    capped_gain = np.clip(returns, 0, gain_cap)
    capped_loss = np.clip(-returns, 0, loss_cap)
    binary_params = {**e1["params"], "seed": SEED}
    regression_params = {**binary_params, "objective": "reg:squarederror", "eval_metric": "rmse"}
    q_flat = _fit_component(x_train, flat.astype(float), ~up, x_valid, binary_params)
    mu_gain = _fit_component(x_train, capped_gain, up, x_valid, regression_params)
    mu_loss = _fit_component(x_train, capped_loss, down, x_valid, regression_params)
    # XGBoost regression outputs are unconstrained. Restrict estimated magnitudes
    # to the nonnegative training-fold target range before combining probabilities.
    q_flat = np.clip(q_flat.astype(float), 0, 1)
    mu_gain = np.clip(mu_gain.astype(float), 0, gain_cap)
    mu_loss = np.clip(mu_loss.astype(float), 0, loss_cap)
    control = saved.loc[saved["fold"].eq(fold.name), KEYS + ["p_up"]]
    paired = valid_rows[KEYS + [OPEN, CLOSE]].merge(control, on=KEYS, how="left",
                                                    validate="one_to_one", indicator=True)
    if len(paired) != len(valid_rows) or not paired["_merge"].eq("both").all():
        raise ValueError(f"Saved E1 stock-date coverage differs on {fold.name}")
    paired = paired.drop(columns="_merge")
    paired = paired.merge(valid[KEYS + ["intraday_return_1d", "taiex_return"]],
                          on=KEYS, validate="one_to_one", how="left", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Validation F0 coverage changed")
    paired = paired.drop(columns="_merge")
    if paired["intraday_return_1d"].isna().any():
        raise ValueError("Previous-direction candidate lacks historical return")
    if not paired["p_up"].between(0, 1).all():
        raise ValueError("Invalid archived up probabilities")
    # Preprocessing and model predictions use valid_rows order; all merges retain
    # left order and the exact key equality is checked again here.
    if not paired[KEYS].equals(valid_rows[KEYS]):
        raise ValueError("Conditional predictions are not aligned to validation keys")
    p_up = paired["p_up"].to_numpy()
    constant_flat = float(flat.sum() / (~up).sum())
    constant_gain = float(np.mean(np.clip(gain, 0, gain_cap)))
    constant_loss = float(np.mean(np.clip(loss, 0, loss_cap)))
    estimated = expected_return(p_up, q_flat, mu_gain, mu_loss)
    constant = expected_return(p_up, np.full(len(paired), constant_flat),
                               np.full(len(paired), constant_gain),
                               np.full(len(paired), constant_loss))
    paired["fold"] = fold.name
    paired["p_up"] = p_up
    paired["p_flat_given_nonup"] = q_flat
    paired["p_down"] = (1-p_up)*(1-q_flat)
    paired["p_flat"] = (1-p_up)*q_flat
    paired["gain_if_up"] = mu_gain
    paired["loss_if_down"] = mu_loss
    paired["estimated_return"] = estimated
    paired["constant_expected_return"] = constant
    paired["e1"] = (p_up > .5).astype(int)
    paired["e3"] = regime_predictions(p_up, paired["taiex_return"].to_numpy(), rule)
    paired["previous_direction_candidate"] = paired["intraday_return_1d"].gt(0).astype(int)
    paired["e11_direction_only"] = paired["e1"]
    paired["e11_constant_payoffs"] = (constant > 0).astype(int)
    paired["e11_conditional_payoffs"] = (estimated > 0).astype(int)
    paired["realized_return"] = (paired[CLOSE]-paired[OPEN])/paired[OPEN]
    paired["actual_up"] = paired["realized_return"].gt(0).astype(int)
    if not np.allclose(paired[["p_up", "p_down", "p_flat"]].sum(axis=1), 1,
                       rtol=0, atol=1e-12):
        raise ValueError("Up/down/flat probabilities do not sum to one")
    for arm in ARMS:
        paired[f"{arm}_contribution"] = (2*paired[arm]-1)*paired["realized_return"]
        result = evaluate_predictions(valid_rows, paired[KEYS + [arm]].rename(
            columns={arm: "predicted_y"}))
        if not np.isclose(result["official_score"], paired[f"{arm}_contribution"].mean(),
                          rtol=0, atol=1e-14):
            raise ValueError(f"Official score mismatch: {fold.name}/{arm}")
    stats = {"fold": fold.name, "training_rows": len(train_rows),
             "training_end": str(train_rows["date"].max().date()),
             "validation_rows": len(valid_rows), "validation_start": str(valid_rows["date"].min().date()),
             "training_up": int(up.sum()), "training_down": int(down.sum()),
             "training_flat": int(flat.sum()),
             "gain_cap": float(gain_cap), "loss_cap": float(loss_cap),
             "gain_clipped_fraction": float((gain > gain_cap).mean()),
             "loss_clipped_fraction": float((loss > loss_cap).mean()),
             "constant_flat_given_nonup": constant_flat,
             "constant_gain": constant_gain, "constant_loss": constant_loss,
             "predicted_gain_floor_fraction": float((mu_gain == 0).mean()),
             "predicted_loss_floor_fraction": float((mu_loss == 0).mean())}
    return paired.sort_values(["date", "stock_id"]).reset_index(drop=True), stats


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, days, distribution, outcomes = [], [], [], []
    for fold, arm, group in ((fold, arm, rows.loc[rows["fold"].eq(fold)])
                             for fold in [f.name for f in FOLDS] for arm in ARMS):
        score = group[f"{arm}_contribution"]
        metrics.append({"fold": fold, "arm": arm, "n": len(group),
                        "official_score": float(score.mean()),
                        "accuracy": float(group[arm].eq(group["actual_up"]).mean()),
                        "predicted_up_rate": float(group[arm].mean())})
        daily = group.groupby("date", sort=True).agg(
            n=(arm,"size"), score_sum=(f"{arm}_contribution", "sum"),
            official_score=(f"{arm}_contribution", "mean"),
            up_rate=(arm,"mean"), actual_up_rate=("actual_up","mean"))
        for date, record in daily.iterrows():
            days.append({"fold": fold, "arm": arm, "date": date, **record.to_dict()})
        distribution.append({"fold": fold, "arm": arm, "n_days": len(daily),
                             "negative_days": int((daily["official_score"] < 0).sum()),
                             "daily_mean": float(daily["official_score"].mean()),
                             "daily_median": float(daily["official_score"].median()),
                             "daily_q10": float(daily["official_score"].quantile(.1)),
                             "daily_q90": float(daily["official_score"].quantile(.9)),
                             "daily_min": float(daily["official_score"].min()),
                             "daily_max": float(daily["official_score"].max())})
        types = np.select([group[arm].eq(1) & group["actual_up"].eq(1),
                           group[arm].eq(0) & group["actual_up"].eq(0),
                           group[arm].eq(1) & group["actual_up"].eq(0)],
                          ["correct_up", "correct_down", "incorrect_up"],
                          default="incorrect_down")
        for category in ("correct_up", "correct_down", "incorrect_up", "incorrect_down"):
            mask = types == category
            outcomes.append({"fold": fold, "arm": arm, "category": category,
                             "n": int(mask.sum()),
                             "sum_contribution": float(score.loc[mask].sum()),
                             "contribution_to_fold_score": float(score.loc[mask].sum()/len(group))})
    return (pd.DataFrame(metrics), pd.DataFrame(days),
            pd.DataFrame(distribution), pd.DataFrame(outcomes))


def _paired_effects(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for fold, group in rows.groupby("fold", sort=True):
        for reference in ("e3", "e11_direction_only", "e11_constant_payoffs"):
            changed = group.loc[group["e11_conditional_payoffs"].ne(group[reference])]
            delta = changed["e11_conditional_payoffs_contribution"] - changed[f"{reference}_contribution"]
            up_changes = changed["e11_conditional_payoffs"].eq(1)
            records.append({"fold": fold, "reference": reference,
                            "changed_rows": len(changed),
                            "down_to_up": int(up_changes.sum()),
                            "up_to_down": int((~up_changes).sum()),
                            "delta_sum": float(delta.sum()),
                            "delta_official_score": float(delta.sum()/len(group)),
                            "beneficial_changes": int((delta > 0).sum()),
                            "harmful_changes": int((delta < 0).sum()),
                            "benefit_sum": float(delta.loc[delta > 0].sum()),
                            "harm_sum": float(delta.loc[delta < 0].sum()),
                            "reference_missed_up_flipped": int((changed[reference].eq(0)
                                & changed["actual_up"].eq(1)).sum())})
    return pd.DataFrame(records)


def _component_validation(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for fold, group in rows.groupby("fold", sort=True):
        up = group["realized_return"].gt(0)
        down = group["realized_return"].lt(0)
        records.append({"fold": fold, "n": len(group),
                        "mean_p_up": float(group["p_up"].mean()),
                        "actual_up_rate": float(up.mean()),
                        "mean_p_flat": float(group["p_flat"].mean()),
                        "actual_flat_rate": float(group["realized_return"].eq(0).mean()),
                        "mean_p_down": float(group["p_down"].mean()),
                        "actual_down_rate": float(down.mean()),
                        "predicted_gain_on_up_rows": float(group.loc[up,"gain_if_up"].mean()),
                        "actual_gain_on_up_rows": float(group.loc[up,"realized_return"].mean()),
                        "predicted_loss_on_down_rows": float(group.loc[down,"loss_if_down"].mean()),
                        "actual_loss_on_down_rows": float(-group.loc[down,"realized_return"].mean()),
                        "mean_estimated_return": float(group["estimated_return"].mean()),
                        "mean_realized_return": float(group["realized_return"].mean())})
    return pd.DataFrame(records)


def _bootstrap(days: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    draws_by_fold = []
    records = []
    for fold in [f.name for f in FOLDS]:
        e3 = days.loc[days["fold"].eq(fold) & days["arm"].eq("e3")].sort_values("date")
        e11 = days.loc[days["fold"].eq(fold) & days["arm"].eq("e11_conditional_payoffs")].sort_values("date")
        if not e3["date"].reset_index(drop=True).equals(e11["date"].reset_index(drop=True)):
            raise ValueError("Paired daily dates differ")
        n = len(e3)
        n_blocks = int(np.ceil(n/BLOCK_DAYS))
        starts = rng.integers(0, n, size=(N_BOOTSTRAP, n_blocks))
        indexes = (starts[:,:,None] + np.arange(BLOCK_DAYS)[None,None,:]) % n
        indexes = indexes.reshape(N_BOOTSTRAP, -1)[:, :n]
        differences = e11["score_sum"].to_numpy() - e3["score_sum"].to_numpy()
        counts = e3["n"].to_numpy()
        draws = differences[indexes].sum(axis=1)/counts[indexes].sum(axis=1)
        observed = differences.sum()/counts.sum()
        draws_by_fold.append(draws)
        records.append({"fold": fold, "observed_e11_minus_e3": float(observed),
                        "ci_2_5": float(np.quantile(draws,.025)),
                        "ci_97_5": float(np.quantile(draws,.975)),
                        "bootstrap_draws": N_BOOTSTRAP, "block_trading_days": BLOCK_DAYS})
    mean_draw = np.mean(draws_by_fold, axis=0)
    records.append({"fold":"equal_year_mean",
                    "observed_e11_minus_e3":float(np.mean([r["observed_e11_minus_e3"] for r in records])),
                    "ci_2_5":float(np.quantile(mean_draw,.025)),
                    "ci_97_5":float(np.quantile(mean_draw,.975)),
                    "bootstrap_draws":N_BOOTSTRAP,"block_trading_days":BLOCK_DAYS})
    return pd.DataFrame(records)


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    saved, e1, rule = _load_controls(f0_path)
    fold_frames, component_stats = [], []
    for fold in FOLDS:
        print(f"Fitting E11 conditional components for {fold.name}", flush=True)
        frame, stats = _fold(fold, market, f0, target_ids, saved, e1, rule)
        fold_frames.append(frame)
        component_stats.append(stats)
    rows = pd.concat(fold_frames, ignore_index=True)
    validate_keys(rows)
    metrics, days, distribution, outcomes = _summaries(rows)
    effects = _paired_effects(rows)
    component_validation = _component_validation(rows)
    uncertainty = _bootstrap(days)
    e3_cv = cv_score(dict(zip(metrics.loc[metrics["arm"].eq("e3"), "fold"],
                               metrics.loc[metrics["arm"].eq("e3"), "official_score"])))
    with (ROOT/"results/zhenci_xgboost_e3_market_regime_v2/summary.json").open(
        encoding="utf-8") as file:
        archived_e3 = json.load(file)
    if not np.isclose(e3_cv, archived_e3["selected_same_period_cv"]["mean_cv_official_score"],
                      rtol=0, atol=1e-12):
        raise ValueError("E3 same-key CV control failed to reproduce archive")
    e1_cv = cv_score(dict(zip(metrics.loc[metrics["arm"].eq("e1"), "fold"],
                               metrics.loc[metrics["arm"].eq("e1"), "official_score"])))
    if not np.isclose(e1_cv, e1["mean_cv_official_score"], rtol=0, atol=1e-12):
        raise ValueError("E1 same-key CV control failed to reproduce archive")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("validation_predictions",rows),("fold_metrics",metrics),
                        ("daily_scores",days),("daily_distributions",distribution),
                        ("score_decomposition",outcomes),("paired_effects",effects),
                        ("paired_uncertainty",uncertainty),
                        ("component_validation",component_validation),
                        ("component_training_stats",pd.DataFrame(component_stats))):
        table.to_csv(output_dir/f"{name}.csv", index=False)
    summary = {"experiment":"E11 conditional expected-return decision",
               "control":"E3 fixed TAIEX threshold on archived E1 full-sample F0 probabilities",
               "reference_seed":SEED,"folds":[f.name for f in FOLDS],
               "arms":list(ARMS),"f0_features":list(F0_FEATURES),
               "component_design":"archived E1 p(up); XGBoost P(flat|not up), E[gain|up], E[loss|down]",
               "decision_formula":"p_up*mu_gain - (1-p_up)*(1-q_flat_given_nonup)*mu_loss > 0",
               "cap_quantile_training_only":CAP_QUANTILE,
               "component_params":e1["params"],"num_boost_round":e1["num_boost_round"],
               "versions":{"python":platform.python_version(),"numpy":np.__version__,
                           "pandas":pd.__version__,"xgboost":xgb.__version__},
               "equal_year_mean_official_scores":{
                   arm:cv_score(dict(zip(metrics.loc[metrics["arm"].eq(arm),"fold"],
                                         metrics.loc[metrics["arm"].eq(arm),"official_score"])))
                   for arm in ARMS},
               "input_sha256":{
                   "f0":sha256(f0_path),
                   "market_100_csv_combined":market_sha256(data_dir),
                   "stock_metadata":sha256(data_dir/"info.json"),
                   "saved_e1_predictions":sha256(ROOT/"results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"),
                   "e1_summary":sha256(ROOT/"results/zheng_xgboost_e1_five_seeds_v1/summary.json"),
                   "e3_summary":sha256(ROOT/"results/zhenci_xgboost_e3_market_regime_v2/summary.json")}}
    with (output_dir/"summary.json").open("w",encoding="utf-8") as file:
        json.dump(summary,file,indent=2)
        file.write("\n")
    print(metrics.pivot(index="fold",columns="arm",values="official_score").to_string())
    print("Equal-year means:",summary["equal_year_mean_official_scores"])
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--f0-path",type=Path,default=ROOT/"data/processed/f0.csv")
    parser.add_argument("--output-dir",type=Path,default=RESULT)
    args = parser.parse_args()
    run(args.data_dir,args.f0_path,args.output_dir)
