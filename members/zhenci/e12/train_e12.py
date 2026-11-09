"""Fixed one-layer LSTM experiment on 20 stock-specific F0 snapshots.

Run from final_project: uv run --extra e12 python -m members.zhenci.e12.train_e12
Only the original 2020-2025 F0/history and archived E1/E3 CV controls are used.
The fixed final epoch is evaluated; outer-year labels never select an epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from members.zhenci.e1.train_e1 import join_f0
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import E1_SEEDS, FOLDS, SEED, split_market

OUTPUT = Path(__file__).resolve().parent / "outputs"
WINDOW = 20
HIDDEN = 32
HEAD_DROPOUT = .2
EPOCHS = 6
BATCH_SIZE = 1024
LEARNING_RATE = .001
BLOCK_DAYS = 5
BOOTSTRAP_DRAWS = 3000
OPEN, CLOSE = "etl:adj_open", "etl:adj_close"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


class StockLSTM(nn.Module):
    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(NUMERIC_FEATURES), hidden_size=HIDDEN,
                            num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(HIDDEN+context_size, 1))

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return self.head(torch.cat((hidden[-1], context), dim=1)).squeeze(1)


def sequence_end_positions(dates: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Indices whose per-stock history contains a complete ordered window."""
    if window < 1 or len(dates) < window:
        return np.empty(0, dtype=np.int64)
    if not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Stock trading sessions must be strictly increasing")
    return np.arange(window-1, len(dates), dtype=np.int64)


def _sequence_data(market: pd.DataFrame, f0: pd.DataFrame, fold, target_ids: list[str],
                   preprocessor: BaselinePreprocessor, train_rows: pd.DataFrame,
                   valid_rows: pd.DataFrame) -> tuple[dict, dict, list[str]]:
    source = market.loc[market["date"].between("2020-01-01", fold.eval_end), KEYS].copy()
    snapshots = join_f0(source, f0, include_label=False).sort_values(
        ["stock_id", "date"]).reset_index(drop=True)
    transformed = preprocessor.transform(snapshots[list(F0_FEATURES)])
    context_names = [field for field in transformed.columns if field not in NUMERIC_FEATURES]
    if not context_names or not all(name.startswith(("stock_id=", "day_of_week="))
                                    for name in context_names):
        raise ValueError("Expected training-fitted stock and weekday context")
    numeric = transformed[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float32)
    context = transformed[context_names].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all() or not np.isfinite(context).all():
        raise ValueError("Nonfinite transformed F0 values")
    buckets = {"train":{"sequences":[],"context":[],"keys":[]},
               "valid":{"sequences":[],"context":[],"keys":[]}}
    target_set = set(target_ids)
    for stock_id, group in snapshots.groupby("stock_id", sort=True):
        positions = group.index.to_numpy()
        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        ends = sequence_end_positions(dates)
        if not len(ends):
            continue
        # sliding_window_view on the row axis yields (samples, features, time).
        windows = np.lib.stride_tricks.sliding_window_view(
            numeric[positions], window_shape=WINDOW, axis=0).transpose(0, 2, 1)
        if windows.shape != (len(ends), WINDOW, len(NUMERIC_FEATURES)):
            raise ValueError("Unexpected F0 sequence dimensions")
        selected_dates = dates[ends]
        choices = {
            "train": selected_dates <= np.datetime64(fold.train_end),
            "valid": ((selected_dates >= np.datetime64(fold.eval_start))
                      & (selected_dates <= np.datetime64(fold.eval_end))
                      & (stock_id in target_set)),
        }
        for split, mask in choices.items():
            if not mask.any():
                continue
            chosen = ends[mask]
            buckets[split]["sequences"].append(np.ascontiguousarray(windows[mask], dtype=np.float32))
            buckets[split]["context"].append(np.ascontiguousarray(context[positions[chosen]], dtype=np.float32))
            buckets[split]["keys"].append(pd.DataFrame({"stock_id": stock_id,
                                                         "date": pd.to_datetime(selected_dates[mask])}))
    complete = {}
    labels = market[KEYS + ["y"]]
    for split in ("train", "valid"):
        bucket = buckets[split]
        if not bucket["keys"]:
            raise ValueError(f"No complete {split} sequences")
        keys = pd.concat(bucket["keys"], ignore_index=True)
        validate_keys(keys)
        labeled = keys.merge(labels, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not labeled["_merge"].eq("both").all() or labeled["y"].isna().any():
            raise ValueError(f"{split} sequence labels or keys missing")
        if not labeled[KEYS].equals(keys):
            raise ValueError(f"{split} sequence order changed on label merge")
        exact = train_rows if split == "train" else valid_rows
        if split == "valid" and (len(keys) != len(exact) or
                set(map(tuple,keys[KEYS].to_numpy())) != set(map(tuple,exact[KEYS].to_numpy()))):
            raise ValueError("Outer validation sequences do not match target keys")
        if split == "train" and not set(map(tuple,keys[KEYS].to_numpy())).issubset(
                set(map(tuple,exact[KEYS].to_numpy()))):
            raise ValueError("Training sequence escaped earlier training window")
        complete[split] = {"keys":keys,
                           "sequences":np.concatenate(bucket["sequences"]),
                           "context":np.concatenate(bucket["context"]),
                           "y":labeled["y"].to_numpy(dtype=np.float32)}
    return complete["train"], complete["valid"], context_names


def _probabilities(model: StockLSTM, data: dict) -> np.ndarray:
    model.eval()
    sequence = torch.from_numpy(data["sequences"])
    context = torch.from_numpy(data["context"])
    output = []
    with torch.no_grad():
        for start in range(0, len(sequence), BATCH_SIZE):
            logits = model(sequence[start:start+BATCH_SIZE], context[start:start+BATCH_SIZE])
            output.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(output)
    if result.shape != (len(data["keys"]),) or not np.isfinite(result).all():
        raise ValueError("Invalid LSTM probability output")
    return result


def _bce(probability: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probability.astype(float), 1e-7, 1-1e-7)
    y = labels.astype(float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def _fit(train: dict, valid: dict, context_size: int, fold_name: str) -> tuple[StockLSTM, pd.DataFrame, float, float, np.ndarray]:
    set_seed()
    model = StockLSTM(context_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(train["sequences"]),
                            torch.from_numpy(train["context"]),
                            torch.from_numpy(train["y"]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    curve = []
    fit_started = time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train()
        for sequence, context, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence, context)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_probability = _probabilities(model, train)
        valid_probability = _probabilities(model, valid)
        curve.append({"fold":fold_name,"epoch":epoch,
                      "train_bce":_bce(train_probability,train["y"]),
                      "outer_validation_bce_diagnostic_only":_bce(valid_probability,valid["y"]),
                      "train_up_rate":float((train_probability>.5).mean()),
                      "outer_validation_up_rate_diagnostic_only":float((valid_probability>.5).mean()),
                      "elapsed_seconds":time.perf_counter()-fit_started})
        print(f"{fold_name} epoch {epoch}/{EPOCHS}: "
              f"train BCE {curve[-1]['train_bce']:.5f}, "
              f"outer validation BCE {curve[-1]['outer_validation_bce_diagnostic_only']:.5f}", flush=True)
    train_seconds = time.perf_counter()-fit_started
    predict_started = time.perf_counter()
    final_probability = _probabilities(model, valid)
    predict_seconds = time.perf_counter()-predict_started
    return model, pd.DataFrame(curve), train_seconds, predict_seconds, final_probability


def _controls(path: Path, e3_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    with (path.parent/"summary.json").open(encoding="utf-8") as file:
        e1_summary = json.load(file)
    with e3_path.open(encoding="utf-8") as file:
        e3_summary = json.load(file)
    if (e1_summary.get("config_id") != "xgboost_initial_v1"
            or e1_summary.get("feature_order") != list(F0_FEATURES)
            or e1_summary.get("seeds") != list(E1_SEEDS)):
        raise ValueError("Archived E1 F0 control changed")
    rule = selected_rule(e3_summary, e1_summary["config_id"])
    saved = pd.read_csv(path,dtype={"stock_id":str,"fold":str},parse_dates=["date"])
    if (set(saved["seed"]) != set(E1_SEEDS)
            or set(saved["fold"]) != {f.name for f in FOLDS}
            or set(saved["config_id"]) != {e1_summary["config_id"]}
            or saved.duplicated(["fold","seed",*KEYS]).any()
            or saved.groupby(["fold",*KEYS])["seed"].nunique().ne(5).any()
            or saved.groupby(["fold",*KEYS])["p_up"].nunique().ne(1).any()):
        raise ValueError("Archived E1 prediction coverage or seed agreement changed")
    return saved.loc[saved["seed"].eq(SEED), ["fold",*KEYS,"p_up","predicted_y"]], e1_summary, rule


def _evaluate(fold, valid_rows: pd.DataFrame, valid: dict, probability: np.ndarray,
              saved: pd.DataFrame, f0: pd.DataFrame, rule: dict) -> tuple[pd.DataFrame,list[dict],pd.DataFrame]:
    lstm = valid["keys"].copy()
    lstm["p_lstm"] = probability.astype(float)
    control = saved.loc[saved["fold"].eq(fold.name), KEYS+["p_up","predicted_y"]]
    paired = valid_rows[KEYS+[OPEN,CLOSE]].merge(lstm,on=KEYS,how="left",
                                                validate="one_to_one",indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("LSTM validation keys differ from official target rows")
    paired = paired.drop(columns="_merge").merge(control,on=KEYS,how="left",
                    validate="one_to_one",indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("E1 control keys differ from LSTM keys")
    paired = paired.drop(columns="_merge").merge(f0[KEYS+["taiex_return"]],
                     on=KEYS,how="left",validate="one_to_one",indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("F0 regime feature keys differ from LSTM keys")
    paired = paired.drop(columns="_merge")
    if not paired["predicted_y"].eq(paired["p_up"].gt(.5).astype(int)).all():
        raise ValueError("Archived E1 labels disagree with probabilities")
    paired["e1"] = paired["predicted_y"].astype(int)
    paired["e3"] = regime_predictions(paired["p_up"].to_numpy(),
                                       paired["taiex_return"].to_numpy(),rule)
    paired["e12"] = paired["p_lstm"].gt(.5).astype(int)
    paired["realized_return"] = (paired[CLOSE]-paired[OPEN])/paired[OPEN]
    paired["actual_up"] = paired["realized_return"].gt(0).astype(int)
    metrics = []
    for arm in ("e1","e3","e12"):
        official = evaluate_predictions(valid_rows,paired[KEYS+[arm]].rename(columns={arm:"predicted_y"}))
        paired[f"{arm}_contribution"] = (2*paired[arm]-1)*paired["realized_return"]
        if not np.isclose(official["official_score"],paired[f"{arm}_contribution"].mean(),
                          rtol=0,atol=1e-14):
            raise ValueError("Official score decomposition failed")
        metrics.append({"fold":fold.name,"arm":arm,**official,
                        "predicted_up_rate":float(paired[arm].mean()),
                        "mean_probability":float(paired["p_lstm"].mean()) if arm=="e12" else
                        float(paired["p_up"].mean())})
    days = []
    for day,group in paired.groupby("date",sort=True):
        for arm in ("e1","e3","e12"):
            days.append({"fold":fold.name,"date":day,"arm":arm,"n":len(group),
                         "score_sum":float(group[f"{arm}_contribution"].sum()),
                         "official_score":float(group[f"{arm}_contribution"].mean()),
                         "predicted_up_rate":float(group[arm].mean()),
                         "actual_up_rate":float(group["actual_up"].mean())})
    paired.insert(0,"fold",fold.name)
    return paired.sort_values(["date","stock_id"]).reset_index(drop=True), metrics, pd.DataFrame(days)


def _bootstrap(days: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records, draws_all = [], []
    for fold in [f.name for f in FOLDS]:
        e3=days.loc[days["fold"].eq(fold)&days["arm"].eq("e3")].sort_values("date")
        e12=days.loc[days["fold"].eq(fold)&days["arm"].eq("e12")].sort_values("date")
        if not e3["date"].reset_index(drop=True).equals(e12["date"].reset_index(drop=True)):
            raise ValueError("Paired trading dates differ")
        n=len(e3)
        starts=rng.integers(0,n,size=(BOOTSTRAP_DRAWS,int(np.ceil(n/BLOCK_DAYS))))
        index=((starts[:,:,None]+np.arange(BLOCK_DAYS)[None,None,:])%n).reshape(BOOTSTRAP_DRAWS,-1)[:,:n]
        delta=e12["score_sum"].to_numpy()-e3["score_sum"].to_numpy()
        count=e3["n"].to_numpy()
        draws=delta[index].sum(axis=1)/count[index].sum(axis=1)
        draws_all.append(draws)
        records.append({"fold":fold,"e12_minus_e3":float(delta.sum()/count.sum()),
                        "ci_2_5":float(np.quantile(draws,.025)),"ci_97_5":float(np.quantile(draws,.975))})
    mean=np.mean(draws_all,axis=0)
    records.append({"fold":"equal_year_mean",
                    "e12_minus_e3":float(np.mean([r["e12_minus_e3"] for r in records])),
                    "ci_2_5":float(np.quantile(mean,.025)),"ci_97_5":float(np.quantile(mean,.975))})
    return pd.DataFrame(records)


def run(data_dir: Path, f0_path: Path, e1_path: Path, e3_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E12 output directory: {output_dir}")
    set_seed()
    target_ids,_=load_metadata(data_dir)
    market=load_market(data_dir)
    f0=load_f0_dataset(f0_path)
    saved,e1_summary,rule=_controls(e1_path,e3_path)
    predictions,metric_rows,daily,curves,runtimes,states=[],[],[],[],[],[]
    for fold in FOLDS:
        train_rows,valid_rows=split_market(market,target_ids,fold)
        if train_rows["date"].max()>=valid_rows["date"].min():
            raise ValueError("Chronological split violated")
        train_f0=join_f0(train_rows,f0,include_label=False)
        preprocessor=BaselinePreprocessor().fit(train_f0[list(F0_FEATURES)])
        build_start=time.perf_counter()
        train,valid,context_names=_sequence_data(market,f0,fold,target_ids,preprocessor,
                                                  train_rows,valid_rows)
        build_seconds=time.perf_counter()-build_start
        print(f"{fold.name}: {len(train['keys'])} training sequences, "
              f"{len(valid['keys'])} target validation sequences",flush=True)
        model,curve,fit_seconds,infer_seconds,probability=_fit(train,valid,len(context_names),fold.name)
        frame,metrics,days=_evaluate(fold,valid_rows,valid,probability,saved,f0,rule)
        predictions.append(frame)
        metric_rows.extend(metrics)
        daily.append(days)
        curves.append(curve)
        runtimes.append({"fold":fold.name,"sequence_build_seconds":build_seconds,
                         "training_and_curve_seconds":fit_seconds,
                         "final_inference_seconds":infer_seconds,
                         "training_sequences":len(train["keys"]),
                         "validation_sequences":len(valid["keys"])})
        states.append((fold.name,{name:value.detach().cpu().clone() for name,value in model.state_dict().items()},
                       preprocessor.to_state(),context_names))
    metrics=pd.DataFrame(metric_rows)
    rows=pd.concat(predictions,ignore_index=True)
    days=pd.concat(daily,ignore_index=True)
    uncertainty=_bootstrap(days)
    e1_cv=cv_score(dict(zip(metrics.loc[metrics["arm"].eq("e1"),"fold"],
                            metrics.loc[metrics["arm"].eq("e1"),"official_score"])))
    e3_cv=cv_score(dict(zip(metrics.loc[metrics["arm"].eq("e3"),"fold"],
                            metrics.loc[metrics["arm"].eq("e3"),"official_score"])))
    with e3_path.open(encoding="utf-8") as file:
        e3_summary=json.load(file)
    if (not np.isclose(e1_cv,e1_summary["mean_cv_official_score"],rtol=0,atol=1e-12)
            or not np.isclose(e3_cv,e3_summary["selected_same_period_cv"]["mean_cv_official_score"],
                              rtol=0,atol=1e-12)):
        raise ValueError("Archived E1/E3 controls failed to reproduce")
    probability_summary=rows.groupby("fold").agg(
        n=("p_lstm","size"),mean_probability=("p_lstm","mean"),
        median_probability=("p_lstm","median"),
        p10_probability=("p_lstm",lambda x:x.quantile(.1)),
        p90_probability=("p_lstm",lambda x:x.quantile(.9)),
        actual_up_rate=("actual_up","mean"),
        e12_up_rate=("e12","mean")).reset_index()
    daily_distribution=days.groupby(["fold","arm"]).agg(
        trading_days=("date","nunique"),negative_days=("official_score",lambda x:(x<0).sum()),
        median_daily_score=("official_score","median"),
        p10_daily_score=("official_score",lambda x:x.quantile(.1)),
        p90_daily_score=("official_score",lambda x:x.quantile(.9)),
        min_daily_score=("official_score","min"),
        max_daily_score=("official_score","max")).reset_index()
    output_dir.mkdir(parents=True,exist_ok=False)
    for name,table in (("validation_predictions",rows),("fold_metrics",metrics),
                       ("daily_scores",days),("daily_distribution",daily_distribution),
                       ("learning_curves",pd.concat(curves,ignore_index=True)),
                       ("runtime",pd.DataFrame(runtimes)),
                       ("probability_summary",probability_summary),
                       ("paired_uncertainty",uncertainty)):
        table.to_csv(output_dir/f"{name}.csv",index=False)
    for fold_name,state,preprocessor_state,context_names in states:
        torch.save(state,output_dir/f"model_{fold_name}.pt")
        with (output_dir/f"preprocessor_{fold_name}.json").open("w",encoding="utf-8") as file:
            json.dump({"state":preprocessor_state,"context_names":context_names},file,indent=2)
            file.write("\n")
    summary={"experiment":"E12 fixed one-layer LSTM sequence model",
             "architecture":{"window":WINDOW,"numeric_features":list(NUMERIC_FEATURES),
                             "context":"prediction-day stock ID and weekday one-hot",
                             "hidden_size":HIDDEN,"layers":1,"head_dropout":HEAD_DROPOUT},
             "training":{"epochs_fixed":EPOCHS,"batch_size":BATCH_SIZE,"loss":"BCEWithLogitsLoss",
                         "optimizer":"Adam","initial_learning_rate":LEARNING_RATE,
                         "seed":SEED,"device":"cpu","gradient_clip_norm":1.0,
                         "outer_validation_epochs":"diagnostic only; final epoch always used"},
             "equal_year_mean_scores":{
                 arm:cv_score(dict(zip(metrics.loc[metrics["arm"].eq(arm),"fold"],
                                       metrics.loc[metrics["arm"].eq(arm),"official_score"])))
                 for arm in ("e1","e3","e12")},
             "input_sha256":{"f0":sha256(f0_path),"e1_predictions":sha256(e1_path),
                             "e1_summary":sha256(e1_path.parent/"summary.json"),
                             "e3_summary":sha256(e3_path)},
             "versions":{"python":platform.python_version(),"numpy":np.__version__,
                         "pandas":pd.__version__,"torch":torch.__version__}}
    with (output_dir/"summary.json").open("w",encoding="utf-8") as file:
        json.dump(summary,file,indent=2)
        file.write("\n")
    print(metrics.pivot(index="fold",columns="arm",values="official_score").to_string())
    print("Equal-year mean:",summary["equal_year_mean_scores"])
    return output_dir


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--f0-path",type=Path,default=ROOT/"data/processed/f0.csv")
    parser.add_argument("--e1-predictions",type=Path,
                        default=ROOT/"results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv")
    parser.add_argument("--e3-summary",type=Path,
                        default=ROOT/"results/zhenci_xgboost_e3_market_regime_v2/summary.json")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    run(args.data_dir,args.f0_path,args.e1_predictions,args.e3_summary,args.output_dir)
