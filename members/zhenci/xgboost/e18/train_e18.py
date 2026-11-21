"""Evaluate and archive the frozen E18 comparison on the 2023–2025 folds."""

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
from members.zhenci.xgboost.e11.train_e11 import training_returns
from members.zhenci.xgboost.e15.train_e15 import BLOCK_DAYS, DRAWS, daily_metrics, interval
from members.zhenci.xgboost.e17.train_e17 import evaluate, fit_classifier
from members.zhenci.xgboost.e18.features_e18 import CATALOGUE, EXTRAS, FAMILIES, ExpandedPreprocessor, build_features, catalogue
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import cv_score
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES
from shared.splits import FOLDS, SEED, evaluation_keys, split_market

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
F0 = DATA / "processed/f0.csv"
OUTPUT = HERE / "outputs"
ARMS = {"E3": (), **{f"E18-{k}": v for k,v in FAMILIES.items()}, "E18-E": EXTRAS}
FEATURE_FAMILY = {feature: family for family, fields in FAMILIES.items() for feature in fields}
SELECTION_POLICY = "Mean above E3, wins in at least two years, worst four-month score at least E3; highest mean wins"


def check_catalogue() -> None:
    if json.loads(CATALOGUE.read_text(encoding="utf-8")) != catalogue():
        raise ValueError("Frozen E18 catalogue differs from the configured definitions")


def diagnosis(frame: pd.DataFrame, fold: str, split: str) -> pd.DataFrame:
    numeric = frame[[*F0_FEATURES,*EXTRAS]].select_dtypes(include="number")
    finite = numeric.replace([np.inf,-np.inf],np.nan)
    distinct = finite.nunique()
    result = pd.DataFrame({"feature":finite.columns,"rows":len(frame),"missing":finite.isna().sum().to_numpy(),
                           "nonfinite_nonmissing":(numeric.notna() & finite.isna()).sum().to_numpy(),
                           "distinct":distinct.to_numpy()})
    result["missing_rate"] = result["missing"]/result["rows"]
    return result.assign(fold=fold,split=split,all_missing=result["distinct"].eq(0),constant=result["distinct"].eq(1),
                         family=result["feature"].map(FEATURE_FAMILY).fillna("F0"))


def correlations(frame: pd.DataFrame, fold: str) -> pd.DataFrame:
    matrix = frame[[*F0_FEATURES,*EXTRAS]].select_dtypes(include="number").corr(min_periods=2)
    i,j = np.where(np.triu(matrix.abs().to_numpy() >= .95,k=1))
    return pd.DataFrame({"fold":fold,"feature_1":matrix.index[i],"feature_2":matrix.columns[j],
                         "pearson":matrix.to_numpy()[i,j]})


def paired_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    sums = daily.pivot(index=["fold","date"],columns="arm",values="contribution_sum")
    counts = daily.pivot(index=["fold","date"],columns="arm",values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"],axis=0).all().all():
        raise ValueError("Paired intervals require identical daily coverage")
    rng = np.random.default_rng(SEED)
    result = []
    for arm in tuple(ARMS)[1:]:
        points,draws = [],[]
        for fold in FOLDS:
            delta = (sums.loc[fold.name,arm]-sums.loc[fold.name,"E3"]).to_numpy()
            n = counts.loc[fold.name,"E3"].to_numpy()
            days = len(n)
            starts = rng.integers(0,days,size=(DRAWS,(days+BLOCK_DAYS-1)//BLOCK_DAYS))
            index = ((starts[...,None]+np.arange(BLOCK_DAYS))%days).reshape(DRAWS,-1)[:,:days]
            points.append(delta.sum()/n.sum())
            draws.append(delta[index].sum(axis=1)/n[index].sum(axis=1))
            result.append(interval(arm,"E3",fold.name,points[-1],draws[-1]))
        result.append(interval(arm,"E3","equal_year",np.mean(points),np.mean(draws,axis=0)))
    return pd.DataFrame(result)


def aggregate_scores(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    result = daily.groupby(["fold","arm",period],as_index=False).agg(
        observations=("n","sum"), days=("date","size"), score_sum=("contribution_sum","sum"),
        market_sum=("market_sum","sum"), selection_sum=("selection_sum","sum"),
        daily_score_sd=("daily_score","std"), negative_day_rate=("daily_score",lambda x:x.lt(0).mean()),
        daily_score_p10=("daily_score",lambda x:x.quantile(.1)),daily_score_p90=("daily_score",lambda x:x.quantile(.9)))
    for total,mean in (("score_sum","official_score"),("market_sum","market_direction"),("selection_sum","stock_selection")):
        result[mean] = result[total]/result["observations"]
    return result


def importance(model, columns: pd.Index, arm: str, fold: str) -> pd.DataFrame:
    gain = model.get_booster().get_score(importance_type="total_gain")
    result = pd.DataFrame({"feature":columns,"total_gain":[gain.get(column,0.0) for column in columns]})
    result["logical_feature"] = result["feature"].str.split("=",n=1).str[0]
    result["family"] = result["logical_feature"].map(FEATURE_FAMILY).fillna("F0")
    return result.assign(fold=fold,arm=arm,gain_fraction=result["total_gain"]/result["total_gain"].sum())


def stability(metrics: pd.DataFrame, monthly: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    result = []
    scores = metrics.pivot(index="fold",columns="arm",values="official_score")
    for arm in ARMS:
        month = monthly.loc[monthly["arm"].eq(arm),"official_score"]
        block = blocks.loc[blocks["arm"].eq(arm)]
        worst = block.loc[block["official_score"].idxmin()]
        result.append({"arm":arm,"equal_year_score":cv_score(scores[arm].to_dict()),
                       "years_beating_e3":int(scores[arm].gt(scores["E3"]).sum()),
                       "worst_four_month_score":worst["official_score"],"worst_four_month_period":worst["block"],
                       "four_month_score_sd":block["official_score"].std(),"monthly_score_sd":month.std(),
                       "monthly_score_p10":month.quantile(.1),"monthly_score_median":month.median(),
                       "monthly_score_p90":month.quantile(.9),"negative_months":int(month.lt(0).sum())})
    table = pd.DataFrame(result)
    control = table.iloc[0]
    table["delta_e3"] = table["equal_year_score"]-control["equal_year_score"]
    table["phase1_supported"] = (table["delta_e3"].gt(0) & table["years_beating_e3"].ge(2)
                                  & table["worst_four_month_score"].ge(control["worst_four_month_score"]))
    return table


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    check_catalogue()
    target_ids, reference_ids = load_metadata(DATA)
    end = FOLDS[-1].eval_end
    market = load_market(DATA).loc[lambda x:x["date"].le(end)]
    f0 = load_f0_dataset(F0).loc[lambda x:x["date"].le(end)]
    if not evaluation_keys(market).equals(evaluation_keys(f0)):
        raise ValueError("F0 and market keys differ")
    began = perf_counter()
    extras, coverage, source_sessions = build_features(market,f0[KEYS],reference_ids)
    build_seconds = perf_counter()-began
    if not evaluation_keys(f0).equals(evaluation_keys(extras)):
        raise ValueError("Expanded panel changed the F0 keys")
    combined = f0.merge(extras[KEYS+list(EXTRAS)],on=KEYS,validate="one_to_one")
    print(f"Built {len(EXTRAS)} fields on {len(combined):,} rows in {build_seconds:.2f}s",flush=True)
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule,e3 = load_fixed_rule(DEFAULT_E3_SUMMARY,recipe["config_id"])
    expected = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    archived = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS,recipe)
    metric_rows,predictions,diagnostics,correlated,gains,run_times = [],[],[],[],[],[]
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="e18-",dir=output.parent) as temporary:
        out = Path(temporary)/"outputs"
        out.mkdir()
        for fold in FOLDS:
            train,valid = split_market(market,target_ids,fold)
            if set(train["stock_id"]) != set(reference_ids):
                raise ValueError(f"Training universe changed in {fold.name}")
            train_x,valid_x = (join_f0(rows,combined,include_label=False) for rows in (train,valid))
            diagnostics.extend(diagnosis(frame,fold.name,split) for split,frame in (("train",train_x),("validation",valid_x)))
            correlated.append(correlations(train_x,fold.name))
            returns = training_returns(valid)
            archived_p = align_saved_probabilities(valid,archived,fold.name,SEED)
            taiex = valid_x["taiex_return"].to_numpy()
            weights = pd.Series(1.0,index=train.index)
            for arm,fields in ARMS.items():
                started = perf_counter()
                processor = ExpandedPreprocessor(fields).fit(train_x)
                x_train,x_valid = (processor.transform(frame) for frame in (train_x,valid_x))
                prepared = perf_counter()
                model = fit_classifier(recipe,x_train,train,weights)
                trained = perf_counter()
                p = model.predict_proba(x_valid)[:,1].astype(float)
                inferred = perf_counter()
                decision = regime_predictions(p,taiex,rule)
                metrics, rows = evaluate(arm,valid,p,decision,returns,fold.name)
                if arm == "E3" and (not np.allclose(p,archived_p,atol=1e-7,rtol=0)
                        or not np.array_equal(decision,regime_predictions(archived_p,taiex,rule))
                        or not np.isclose(metrics["official_score"],expected["fold_official_scores_by_seed"][str(SEED)][fold.name],atol=1e-12,rtol=0)):
                    raise ValueError(f"Retrained E3 failed archived control verification in {fold.name}")
                metric_rows.append({**metrics,"logical_fields":len(F0_FEATURES)+len(fields),"encoded_fields":x_train.shape[1]})
                predictions.append(rows)
                gains.append(importance(model,x_train.columns,arm,fold.name))
                run_times.append({"fold":fold.name,"arm":arm,"training_rows":len(train),"preprocessing_seconds":prepared-started,
                                  "training_seconds":trained-prepared,"inference_seconds":inferred-trained})
                artifact = out/"models"/fold.name/arm
                artifact.mkdir(parents=True)
                model.get_booster().save_model(str(artifact/"model.json"))
                (artifact/"preprocessing.json").write_text(json.dumps(processor.state(),indent=2,allow_nan=False),encoding="utf-8")
                print(f"{fold.name} {arm}: {metrics['official_score']:+.8f}",flush=True)

        paired = pd.concat(predictions,ignore_index=True)
        daily = daily_metrics(paired)
        groups = paired.assign(position=2*paired["predicted_y"]-1).groupby(["fold","arm","date"],as_index=False)
        signed = groups.agg(mean_position=("position","mean"),mean_realized_return=("realized_return","mean"))
        daily = daily.merge(signed,on=["fold","arm","date"],validate="one_to_one")
        daily["market_direction"] = daily["mean_position"]*daily["mean_realized_return"]
        daily["stock_selection"] = daily["daily_score"]-daily["market_direction"]
        daily["market_sum"] = daily["market_direction"] * daily["n"]
        daily["selection_sum"] = daily["stock_selection"] * daily["n"]
        daily["month"] = daily["date"].dt.strftime("%Y-%m")
        daily["block"] = daily["fold"]+"-B"+((daily["date"].dt.month-1)//4+1).astype(str)
        monthly,blocks = (aggregate_scores(daily,period) for period in ("month","block"))
        daily_summary = daily.groupby(["fold","arm"]).agg(
            mean_spearman=("spearman","mean"),defined_spearman_days=("spearman","count"),
            daily_score_mean=("daily_score","mean"),daily_score_std=("daily_score","std"),
            negative_day_rate=("daily_score",lambda x:x.lt(0).mean()),daily_score_p10=("daily_score",lambda x:x.quantile(.1)),
            daily_score_median=("daily_score","median"),daily_score_p90=("daily_score",lambda x:x.quantile(.9)),
            market_sum=("market_sum","sum"),selection_sum=("selection_sum","sum"))
        metrics = pd.DataFrame(metric_rows).join(daily_summary,on=["fold","arm"])
        metrics["market_direction"] = metrics.pop("market_sum")/metrics["n_eval"]
        metrics["stock_selection"] = metrics.pop("selection_sum")/metrics["n_eval"]
        if not np.allclose(metrics["official_score"],metrics["market_direction"]+metrics["stock_selection"],atol=1e-12,rtol=0):
            raise ValueError("Score contributions do not reconstruct the official score")
        control = metrics.loc[metrics["arm"].eq("E3")].set_index("fold")["official_score"]
        metrics["delta_e3"] = metrics["official_score"]-metrics["fold"].map(control)
        comparison = stability(metrics,monthly,blocks)
        supported = comparison.loc[comparison["phase1_supported"]].sort_values("equal_year_score",ascending=False)
        selected = None if supported.empty else supported.iloc[0]["arm"]
        gain = pd.concat(gains,ignore_index=True)
        tables = {"fold_metrics":metrics,"stability":comparison,"daily_scores":daily,"monthly_scores":monthly,
                  "four_month_scores":blocks,"validation_predictions":paired,"paired_uncertainty":paired_intervals(daily),
                  "feature_diagnostics":pd.concat(diagnostics),"high_correlations":pd.concat(correlated),
                  "feature_importance":gain,"family_importance":gain.groupby(["fold","arm","family"],as_index=False)["gain_fraction"].sum(),
                  "runtime":pd.DataFrame(run_times),"market_coverage":coverage,"stock_source_sessions":source_sessions,
                  "feature_provenance":extras.drop(columns=list(EXTRAS))}
        for name,table in tables.items():
            table.to_csv(out/f"{name}.csv",index=False)
        paths = (F0,DATA/"info.json",CATALOGUE,HERE/"feasibility_audit.md",HERE/"features_e18.py",Path(__file__),
                 DEFAULT_E1_SUMMARY,DEFAULT_E1_PREDICTIONS,DEFAULT_E3_SUMMARY,ROOT/"shared/features.py",
                 ROOT/"shared/preprocessing.py",ROOT/"shared/evaluation.py")
        summary = {"selected_candidate":selected,"selection_policy":SELECTION_POLICY,"arm_extras":ARMS,
                   "params":recipe["params"],"rounds":recipe["num_boost_round"],"seed":SEED,"taiex_rule":rule,
                   "feature_build_seconds":build_seconds,"block_days":BLOCK_DAYS,"bootstrap_draws":DRAWS,
                   "input_sha256":{path.relative_to(ROOT).as_posix():_sha256(path) for path in paths},
                   "market_sha256":{stock:_sha256(DATA/"market"/f"{stock}.csv") for stock in reference_ids},
                   "versions":{"python":platform.python_version(),"xgboost":xgb.__version__,"scikit_learn":sklearn.__version__,
                               "numpy":np.__version__,"pandas":pd.__version__}}
        (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        out.rename(output)
    print(comparison.to_string(index=False),flush=True)
    print(f"Selected Phase 1 candidate: {selected or 'none'}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
