"""Fit and freeze an exploratory E11 artifact on 2020-2025 history.

Run from final_project: uv run python -m members.zhenci.e11.save_e11
This reuses the existing frozen E1 up-probability model and preprocessing.
It trains only E11's flat, gain, and loss components. No holdout or Phase 1
labels are used, and this script does not create or upload a submission.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.save_e1 import _sha256, holdout_training_rows
from members.zhenci.e1.train_e1 import join_f0
from members.zhenci.e11.train_e11 import CAP_QUANTILE, SEED, market_sha256, training_returns
from shared.data_loader import load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import HOLDOUT

DEFAULT_E1_ARTIFACT = ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
DEFAULT_E11_CV_SUMMARY = ROOT / "members/zhenci/e11/outputs/summary.json"
DEFAULT_ARTIFACT = ROOT / "results/zhenci_e11_expected_return_pre_phase1_artifact"


def verify_source_artifact(source: Path, f0_path: Path) -> tuple[dict, BaselinePreprocessor]:
    with (source/"manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("artifact_version") != 1
            or manifest.get("config_id") != "xgboost_initial_v1"
            or manifest.get("train_start") != "2020-01-01"
            or manifest.get("train_end") != HOLDOUT.train_end
            or manifest.get("training_stocks") != 100
            or manifest.get("seed") != SEED
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("threshold") != .5
            or manifest.get("num_boost_round") != 100
            or manifest.get("f0_csv_sha256") != _sha256(f0_path)):
        raise ValueError("Source E1 artifact is not the frozen pre-holdout F0 control")
    for name, field in (("model.json", "model_sha256"),
                        ("preprocessor.json", "preprocessor_sha256"),
                        ("feature_definitions.py", "feature_definitions_sha256"),
                        ("preprocessing_definitions.py", "preprocessing_definitions_sha256")):
        if _sha256(source/name) != manifest.get(field):
            raise ValueError(f"Source E1 artifact changed: {name}")
    if (_sha256(ROOT/"shared/features.py") != manifest["feature_definitions_sha256"]
            or _sha256(ROOT/"shared/preprocessing.py") != manifest["preprocessing_definitions_sha256"]):
        raise ValueError("Shared F0 definitions no longer match frozen E1 artifact")
    with (source/"preprocessor.json").open(encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    if list(preprocessor.feature_names_) != manifest.get("transformed_feature_order"):
        raise ValueError("Source transformed feature order changed")
    return manifest, preprocessor


def verify_e11_cv(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        summary = json.load(file)
    scores = summary.get("equal_year_mean_official_scores", {})
    if (summary.get("f0_features") != list(F0_FEATURES)
            or summary.get("reference_seed") != SEED
            or summary.get("cap_quantile_training_only") != CAP_QUANTILE
            or summary.get("num_boost_round") != 100
            or set(scores) != {"e1", "e3", "previous_direction_candidate",
                               "e11_direction_only", "e11_constant_payoffs",
                               "e11_conditional_payoffs"}):
        raise ValueError("E11 validation summary is incompatible with deployment recipe")
    return summary


def save_e11(source_artifact: Path, cv_summary_path: Path, data_dir: Path,
             f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E11 artifact directory: {output_dir}")
    e1_manifest, preprocessor = verify_source_artifact(source_artifact, f0_path)
    e11_cv = verify_e11_cv(cv_summary_path)
    if e11_cv.get("component_params") != {k:v for k,v in e1_manifest["params"].items() if k != "seed"}:
        raise ValueError("E11 CV component settings differ from frozen E1 settings")
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    train_rows = holdout_training_rows(market, target_ids, all_ids)
    if (len(train_rows) != e1_manifest["training_rows"]
            or train_rows["date"].max() > pd.Timestamp(HOLDOUT.train_end)):
        raise ValueError("E11 training rows differ from the E1 frozen training window")
    train = join_f0(train_rows, load_f0_dataset(f0_path), include_label=True)
    x_train = preprocessor.transform(train[list(F0_FEATURES)])
    if x_train.columns.tolist() != e1_manifest["transformed_feature_order"]:
        raise ValueError("Frozen preprocessing feature order differs")
    returns = training_returns(train_rows)
    up, down, flat = returns > 0, returns < 0, returns == 0
    gain, loss = returns[up], -returns[down]
    if min(up.sum(), down.sum(), flat.sum()) < 100:
        raise ValueError("Insufficient conditional training observations")
    gain_cap = float(np.quantile(gain, CAP_QUANTILE))
    loss_cap = float(np.quantile(loss, CAP_QUANTILE))
    if not (np.isfinite(gain_cap) and np.isfinite(loss_cap) and gain_cap > 0 and loss_cap > 0):
        raise ValueError("Invalid conditional magnitude cap")
    base_params = dict(e1_manifest["params"])
    binary_params = {**base_params, "objective":"binary:logistic", "eval_metric":"logloss"}
    reg_params = {**base_params, "objective":"reg:squarederror", "eval_metric":"rmse"}
    specs = (
        ("flat", ~up, flat.astype(float), binary_params),
        ("gain", up, np.clip(returns, 0, gain_cap), reg_params),
        ("loss", down, np.clip(-returns, 0, loss_cap), reg_params),
    )
    boosters = {}
    for name, mask, labels, params in specs:
        if len(np.unique(labels[mask])) < 2:
            raise ValueError(f"Constant E11 {name} target")
        print(f"Fitting E11 {name} component on {int(mask.sum())} rows", flush=True)
        boosters[name] = xgb.train(params, xgb.DMatrix(x_train.loc[mask], label=labels[mask]),
                                   num_boost_round=e1_manifest["num_boost_round"])
    output_dir.mkdir(parents=True, exist_ok=False)
    copies = {"model_up.json":"model.json", "preprocessor.json":"preprocessor.json",
              "feature_definitions.py":"feature_definitions.py",
              "preprocessing_definitions.py":"preprocessing_definitions.py"}
    for destination, original in copies.items():
        shutil.copyfile(source_artifact/original, output_dir/destination)
    for name, booster in boosters.items():
        booster.save_model(str(output_dir/f"model_{name}.json"))
    # Verify that loading every saved model preserves a sample of predictions.
    probe = xgb.DMatrix(x_train.iloc[:5])
    source_up = xgb.Booster(model_file=str(source_artifact/"model.json"))
    copied_up = xgb.Booster(model_file=str(output_dir/"model_up.json"))
    if not np.array_equal(source_up.predict(probe), copied_up.predict(probe)):
        raise ValueError("Copied E1 direction model changed predictions")
    for name, booster in boosters.items():
        restored = xgb.Booster(model_file=str(output_dir/f"model_{name}.json"))
        if not np.array_equal(booster.predict(probe), restored.predict(probe)):
            raise ValueError(f"Saved E11 {name} model changed predictions")
    files = (*copies.keys(), *(f"model_{name}.json" for name in boosters))
    manifest = {
        "artifact_version":1,
        "status":"exploratory_e11_pre_phase1_requested",
        "reason":"E11 did not consistently beat E3 in historical CV; user requested an exploratory submission",
        "train_start":"2020-01-01", "train_end":HOLDOUT.train_end,
        "training_rows":len(train_rows), "training_stocks":train_rows["stock_id"].nunique(),
        "training_up":int(up.sum()),"training_down":int(down.sum()),"training_flat":int(flat.sum()),
        "seed":SEED,"feature_order":list(F0_FEATURES),
        "transformed_feature_order":list(preprocessor.feature_names_),
        "formula":"p_up*mu_gain - (1-p_up)*(1-q_flat_given_nonup)*mu_loss > 0",
        "direction_model":"copied frozen full-sample E1 XGBoost model",
        "component_params":{"flat":binary_params,"gain":reg_params,"loss":reg_params},
        "num_boost_round":e1_manifest["num_boost_round"],
        "cap_quantile_training_only":CAP_QUANTILE,
        "gain_cap":gain_cap,"loss_cap":loss_cap,
        "gain_clipped_fraction":float((gain > gain_cap).mean()),
        "loss_clipped_fraction":float((loss > loss_cap).mean()),
        "source_e1_manifest_sha256":_sha256(source_artifact/"manifest.json"),
        "e11_cv_summary_sha256":_sha256(cv_summary_path),
        "f0_csv_sha256":_sha256(f0_path),
        "market_csv_combined_sha256":market_sha256(data_dir),
        "stock_metadata_sha256":_sha256(data_dir/"info.json"),
        "file_sha256":{name:_sha256(output_dir/name) for name in files},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,
                    "pandas":pd.__version__,"xgboost":xgb.__version__},
    }
    with (output_dir/"manifest.json").open("w",encoding="utf-8") as file:
        json.dump(manifest,file,indent=2,allow_nan=False)
        file.write("\n")
    print(f"Exploratory E11 artifact saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-e1-artifact",type=Path,default=DEFAULT_E1_ARTIFACT)
    parser.add_argument("--e11-cv-summary",type=Path,default=DEFAULT_E11_CV_SUMMARY)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--f0-path",type=Path,default=ROOT/"data/processed/f0.csv")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    save_e11(args.source_e1_artifact,args.e11_cv_summary,args.data_dir,args.f0_path,args.output_dir)
