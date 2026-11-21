"""Inspect frozen E3 trees and evaluate prefix predictions without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e1.save_e1 import _sha256, holdout_training_rows
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY,
    align_saved_probabilities, load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import CLOSE, OPEN, cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import FOLDS, SEED, split_market

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FOLD_SOURCE = ROOT / "members/zhenci/xgboost/e18/outputs"
FINAL_SOURCE = ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
F0 = ROOT / "data/processed/f0.csv"
FRACTIONS = (10, 25, 50, 75, 100)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoints(rounds: int) -> tuple[set[int], list[int]]:
    # Integer half-up rounding, clamped to at least one round.
    primary = {max(1, (rounds * pct + 50) // 100) for pct in FRACTIONS}
    return primary, sorted(primary | {1} | set(range(5, rounds, 5)))


def distribution(split: str, labels: np.ndarray, p: np.ndarray, calls: np.ndarray) -> dict:
    p = p.astype(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite logistic probabilities")
    q = np.clip(p, 1e-15, 1 - 1e-15)
    values = {"bce": -np.mean(np.where(labels == 1, np.log(q), np.log1p(-q))),
              "auc": roc_auc_score(labels, p), "accuracy": np.mean(calls == labels),
              "accuracy_050": np.mean((p > .5) == labels), "brier": np.mean((p - labels)**2),
              "predicted_up_rate": calls.mean(), "actual_up_rate": labels.mean(),
              "p_mean": p.mean(), "p_sd": p.std(), "p_lt_010": (p < .1).mean(), "p_gt_090": (p > .9).mean()}
    values.update({f"p{pct:02d}": np.quantile(p, pct / 100) for pct in (1, 10, 50, 90, 99)})
    return {f"{split}_{key}": value for key, value in values.items()}


def daily_stats(frame: pd.DataFrame) -> pd.DataFrame:
    tagged = frame[KEYS + ["p_up", "realized_return", "contribution"]].copy()
    ranks = tagged.groupby("date")[["p_up","realized_return"]].rank(method="average")
    centered = ranks - ranks.groupby(tagged["date"]).transform("mean")
    tagged["cross"] = centered["p_up"] * centered["realized_return"]
    tagged["p_sq"] = centered["p_up"]**2
    tagged["r_sq"] = centered["realized_return"]**2
    out = tagged.groupby("date",as_index=False).agg(n=("contribution","size"),
        score_sum=("contribution","sum"), cross=("cross","sum"), p_sq=("p_sq","sum"), r_sq=("r_sq","sum"))
    out["daily_score"] = out["score_sum"] / out["n"]
    variance = out.pop("p_sq") * out.pop("r_sq")
    out["spearman"] = out.pop("cross") / np.sqrt(variance.where(variance.gt(0)))
    return out


def periodic(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    result = daily.groupby(["fold", "rounds", key],as_index=False).agg(
        rows=("n","sum"), days=("date","size"), score_sum=("score_sum","sum"),
        daily_sd=("daily_score", "std"), negative_day_rate=("daily_score",lambda x:x.lt(0).mean()),
        mean_daily_spearman=("spearman","mean"))
    result["official_score"] = result["score_sum"] / result["rows"]
    return result


def inspect_trees(model: xgb.Booster, train: xgb.DMatrix, validation: xgb.DMatrix | None,
                  label: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    learned = model.get_dump(dump_format="json",with_stats=True)
    n = model.num_boosted_rounds()
    if len(learned) != n:
        raise ValueError("Expected one tree per round in binary E3")
    seen_train = model.predict(train,pred_leaf=True,iteration_range=(0,n)).astype(int)
    seen_valid = None if validation is None else model.predict(validation,pred_leaf=True,iteration_range=(0,n)).astype(int)
    trees, leaves = [], []
    for i, tree in enumerate(learned):
        root = json.loads(tree)
        stack = [(root,0)]
        train_counts = np.bincount(seen_train[:,i])
        valid_counts = None if seen_valid is None else np.bincount(seen_valid[:,i])
        group = []
        splits = 0
        while stack:
            node, depth = stack.pop()
            if "children" in node:
                splits += 1
                stack.extend((child,depth+1) for child in node["children"])
                continue
            leaf = node["nodeid"]
            group.append({"model":label,"tree":i+1,"leaf":leaf,"depth":depth,"leaf_value":node["leaf"],
                          "hessian_cover":node["cover"],"train_rows":int(train_counts[leaf]) if leaf<len(train_counts) else 0,
                          "validation_rows":(np.nan if valid_counts is None else int(valid_counts[leaf]) if leaf<len(valid_counts) else 0)})
        leaves.extend(group)
        if (sum(row["train_rows"] for row in group) != train.num_row()
                or (validation is not None and sum(row["validation_rows"] for row in group) != validation.num_row())):
            raise ValueError("Leaf routing does not account for every row")
        trees.append({"model":label,"tree":i+1,"max_depth":max(row["depth"] for row in group),
                      "leaves":len(group),"splits":splits,"root_hessian_cover":root["cover"],
                      "train_empty_leaves":sum(row["train_rows"]==0 for row in group),
                      "validation_empty_leaves":np.nan if validation is None else sum(row["validation_rows"]==0 for row in group)})
    return pd.DataFrame(trees),pd.DataFrame(leaves)


def predictor_matrix(rows: pd.DataFrame, f0: pd.DataFrame, pre: BaselinePreprocessor) -> tuple[xgb.DMatrix,np.ndarray]:
    features = join_f0(rows,f0,include_label=False)
    transformed = pre.transform(features[list(F0_FEATURES)])
    return xgb.DMatrix(transformed,nthread=4),features["taiex_return"].to_numpy()


def archived_search() -> tuple[pd.DataFrame,pd.DataFrame,list[Path]]:
    paths = [ROOT / "results" / name / "summary.json" for name in (
        "zhenci_xgboost_e2_grid_v1","zhenci_xgboost_e3_optuna_v1","zhenci_xgboost_e3_dart_v1",
        "zhenci_xgboost_e6_nested_stop_v1")]
    trials = []
    for path, key in zip(paths,("configurations","trials","comparisons","comparisons")):
        summary = load_json(path)
        for row in summary[key]:
            param = row.get("params",{})
            trials.append({"study":path.parent.name,"configuration":row.get("arm",row["config_id"]),
                           "mean_cv_score":row["mean_cv_official_score"],
                           "rounds":row.get("num_boost_round",param.get("num_boost_round")),
                           "params_json":json.dumps(param,sort_keys=True) if param else None})
    rounds_path = paths[-1].with_name("selected_rounds.csv")
    return pd.DataFrame(trials),pd.read_csv(rounds_path),[*paths,rounds_path]


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule,e3 = load_fixed_rule(DEFAULT_E3_SUMMARY,recipe["config_id"])
    expected = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    saved = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS,recipe)
    source = load_json(FOLD_SOURCE/"summary.json")
    if (source["params"] != recipe["params"] or source["seed"] != SEED or source["arm_extras"]["E3"]
            or source["input_sha256"]["data/processed/f0.csv"] != _sha256(F0)):
        raise ValueError("Frozen E3 fold bundle does not match the original recipe and F0")
    data = ROOT/"data"
    target_ids,all_ids = load_metadata(data)
    market = load_market(data)
    f0 = load_f0_dataset(F0).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    metrics,daily,predictions,trees,leaves = [],[],[],[],[]
    files = [F0,DEFAULT_E1_SUMMARY,DEFAULT_E1_PREDICTIONS,DEFAULT_E3_SUMMARY,FOLD_SOURCE/"summary.json",Path(__file__)]
    for fold in FOLDS:
        train,valid = split_market(market,target_ids,fold)
        if set(train["stock_id"]) != set(all_ids):
            raise ValueError("Frozen training universe changed")
        folder = FOLD_SOURCE/"models"/fold.name/"E3"
        state_path,model_path = folder/"preprocessing.json",folder/"model.json"
        state = load_json(state_path)
        if state["extras"]:
            raise ValueError("Expected the plain F0 control")
        pre = BaselinePreprocessor.from_state(state["base"])
        model = xgb.Booster(model_file=str(model_path))
        model.set_param({"nthread":4})
        dtrain,tai_train = predictor_matrix(train,f0,pre)
        dvalid,tai_valid = predictor_matrix(valid,f0,pre)
        n = model.num_boosted_rounds()
        if n != recipe["num_boost_round"]:
            raise ValueError("Frozen model contains an unexpected number of rounds")
        primary,round_counts = checkpoints(n)
        train_y,valid_y = train["y"].to_numpy(),valid["y"].to_numpy()
        is_target = train["stock_id"].isin(target_ids).to_numpy()
        realized = ((valid[CLOSE]-valid[OPEN])/valid[OPEN]).to_numpy()
        if not np.array_equal(realized>0,valid_y):
            raise ValueError("Validation labels do not match the official return direction")
        archive = align_saved_probabilities(valid,saved,fold.name,SEED)
        for k in round_counts:
            p_train = model.predict(dtrain,iteration_range=(0,k))
            p_valid = model.predict(dvalid,iteration_range=(0,k))
            train_calls = regime_predictions(p_train,tai_train,rule)
            valid_calls = regime_predictions(p_valid,tai_valid,rule)
            frame = valid[KEYS].assign(p_up=p_valid,actual_up=valid_y,predicted_y=valid_calls,
                realized_return=realized,contribution=(2*valid_calls-1)*realized)
            official = evaluate_predictions(valid,frame[KEYS+["predicted_y"]])
            error = np.nan
            if k == n:
                error = np.max(np.abs(p_valid-archive))
                if (error > 1e-7 or not np.array_equal(valid_calls,regime_predictions(archive,tai_valid,rule))
                        or not np.isclose(official["official_score"],expected["fold_official_scores_by_seed"][str(SEED)][fold.name],atol=1e-12,rtol=0)):
                    raise ValueError(f"100% archived E3 predictions do not reproduce for {fold.name}")
            days = daily_stats(frame).assign(fold=fold.name,rounds=k)
            daily.append(days)
            metrics.append({"fold":fold.name,"rounds":k,"primary_checkpoint":k in primary,
                "train_n":len(train),"validation_n":len(valid),"validation_official_score":official["official_score"],
                "validation_mean_daily_spearman":days["spearman"].mean(),"spearman_days":days["spearman"].count(),
                "validation_daily_score_sd":days["daily_score"].std(),"validation_negative_day_rate":days["daily_score"].lt(0).mean(),
                "max_archived_probability_error":error,
                **distribution("train",train_y,p_train,train_calls),
                **distribution("train_top50",train_y[is_target],p_train[is_target],train_calls[is_target]),
                **distribution("validation",valid_y,p_valid,valid_calls)})
            if k in primary:
                predictions.append(frame.assign(fold=fold.name,rounds=k))
        t,l = inspect_trees(model,dtrain,dvalid,fold.name)
        trees.append(t)
        leaves.append(l)
        files.extend([state_path,model_path])
        print(f"{fold.name}: verified archived probabilities after {n} rounds; measured {len(round_counts)} checkpoints",flush=True)

    manifest = load_json(FINAL_SOURCE/"manifest.json")
    if (manifest["train_end"] != FOLDS[-1].eval_end or manifest["seed"] != SEED
            or manifest["model_sha256"] != _sha256(FINAL_SOURCE/"model.json")
            or manifest["preprocessor_sha256"] != _sha256(FINAL_SOURCE/"preprocessor.json")):
        raise ValueError("Unexpected final E3 source artifact")
    final_train = holdout_training_rows(market,target_ids,all_ids)
    if len(final_train) != manifest["training_rows"]:
        raise ValueError("Final E3 fitting row count changed")
    final_pre = BaselinePreprocessor.from_state(load_json(FINAL_SOURCE/"preprocessor.json"))
    final = xgb.Booster(model_file=str(FINAL_SOURCE/"model.json"))
    final.set_param({"nthread":4})
    dtrain,_ = predictor_matrix(final_train,f0,final_pre)
    t,l = inspect_trees(final,dtrain,None,"final_2020_2025")
    trees.append(t)
    leaves.append(l)
    files.extend(FINAL_SOURCE/name for name in ("manifest.json","model.json","preprocessor.json"))
    search,e6,prior_files = archived_search()
    files.extend(prior_files)
    tree_table,leaf_table = pd.concat(trees,ignore_index=True),pd.concat(leaves,ignore_index=True)
    metrics = pd.DataFrame(metrics)
    daily = pd.concat(daily,ignore_index=True)
    daily["month"] = daily["date"].dt.strftime("%Y-%m")
    daily["block"] = daily["fold"]+"-B"+((daily["date"].dt.month-1)//4+1).astype(str)
    months,blocks = (periodic(daily,key) for key in ("month","block"))
    for tag,table in (("monthly",months),("four_month",blocks)):
        stability = table.groupby(["fold","rounds"])["official_score"].agg(
            **{f"{tag}_min":"min",f"{tag}_sd":"std",f"{tag}_negative_periods":lambda x:x.lt(0).sum()})
        metrics = metrics.join(stability,on=["fold","rounds"])
    curve = metrics.groupby("rounds",as_index=False).agg(
        mean_train_bce=("train_bce","mean"),mean_validation_bce=("validation_bce","mean"),
        mean_train_auc=("train_auc","mean"),mean_validation_auc=("validation_auc","mean"),
        mean_train_accuracy=("train_accuracy","mean"),mean_validation_accuracy=("validation_accuracy","mean"),
        mean_validation_brier=("validation_brier","mean"),mean_validation_spearman=("validation_mean_daily_spearman","mean"),
        mean_validation_up_rate=("validation_predicted_up_rate","mean"))
    series = {k:cv_score(dict(zip(group["fold"],group["validation_official_score"]))) for k,group in metrics.groupby("rounds")}
    curve["equal_year_official_score"] = curve["rounds"].map(series)
    output.mkdir(parents=True,exist_ok=False)
    tables = {"checkpoint_metrics":metrics,"equal_year_curves":curve,"daily_scores":daily,"monthly_scores":months,
              "four_month_scores":blocks,"validation_predictions":pd.concat(predictions,ignore_index=True),
              "trees":tree_table,"leaves":leaf_table,"prior_search":search,"e6_selected_rounds":e6}
    for name,table in tables.items():
        table.to_csv(output/f"{name}.csv",index=False)
    summary = {"xgboost_version":xgb.__version__,"seed":SEED,"base_recipe":recipe["params"],"decision_rule":rule,
               "primary_fractions_percent":FRACTIONS,"primary_rounds":sorted(checkpoints(recipe["num_boost_round"])[0]),
               "checkpoint_rounding":"half-up and at least one tree","final_artifact_train_end":manifest["train_end"],
               "input_sha256":{str(path.relative_to(ROOT)):_sha256(path) for path in files}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(curve.loc[curve["rounds"].isin(summary["primary_rounds"])].to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
