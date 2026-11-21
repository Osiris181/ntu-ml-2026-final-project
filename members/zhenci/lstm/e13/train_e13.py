"""E13: compare E12 LSTM windows 5, 10, 20 on the original historical folds.

Run from final_project: uv run --extra e12 python -m members.zhenci.lstm.e13.train_e13
The archived E12 20-session/seed-42 control is reused, not retrained.
Completed run folders are verified and reused when resuming an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from members.zhenci._archived_sources import archived_source_hash
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.lstm.e12.train_e12 import (
    BATCH_SIZE, EPOCHS, HEAD_DROPOUT, HIDDEN, LEARNING_RATE,
    StockLSTM, _bce, _probabilities, _sequence_data, set_seed, sha256,
)
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import FOLDS, SEED, split_market

OUTPUT = Path(__file__).resolve().parent / "outputs"
E12_OUTPUT = ROOT / "members/zhenci/lstm/e12/outputs"
F0_PATH = ROOT / "data/processed/f0.csv"
WINDOWS = (5, 10, 20)
SEEDS = (42, 43, 44)


def slice_window(batch: dict, window: int) -> dict:
    """Use the identical 20-session sample keys for every candidate window."""
    if window not in WINDOWS or batch["sequences"].shape[1:] != (20, len(NUMERIC_FEATURES)):
        raise ValueError("E13 requires an E12 20-session source batch")
    return {**batch, "sequences": np.ascontiguousarray(batch["sequences"][:, -window:, :])}


def fit_candidate(train: dict, valid: dict, context_size: int, fold: str,
                  window: int, seed: int) -> tuple[StockLSTM, pd.DataFrame, np.ndarray, float, float]:
    """E12 training loop, with only window and random seed varied."""
    set_seed(seed)
    model = StockLSTM(context_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(train["sequences"]),
                            torch.from_numpy(train["context"]),
                            torch.from_numpy(train["y"]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    curve = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for sequence, context, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(sequence, context), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        p_train = _probabilities(model, train)
        p_valid = _probabilities(model, valid)
        curve.append({"fold": fold, "window": window, "seed": seed, "epoch": epoch,
                      "train_bce": _bce(p_train, train["y"]),
                      "outer_validation_bce_diagnostic_only": _bce(p_valid, valid["y"]),
                      "train_up_rate": float((p_train > .5).mean()),
                      "outer_validation_up_rate_diagnostic_only": float((p_valid > .5).mean()),
                      "elapsed_seconds": time.perf_counter() - started})
        print(f"window={window} seed={seed} fold={fold} epoch={epoch}/{EPOCHS} "
              f"train BCE={curve[-1]['train_bce']:.5f} "
              f"valid BCE={curve[-1]['outer_validation_bce_diagnostic_only']:.5f}", flush=True)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    final_p = _probabilities(model, valid)
    inference_seconds = time.perf_counter() - started
    return model, pd.DataFrame(curve), final_p, train_seconds, inference_seconds


def input_hashes() -> dict[str, str]:
    paths = {"original_f0": F0_PATH,
             "e12_summary": E12_OUTPUT / "summary.json",
             "e12_predictions": E12_OUTPUT / "validation_predictions.csv",
             "e12_curves": E12_OUTPUT / "learning_curves.csv",
             "e12_metrics": E12_OUTPUT / "fold_metrics.csv",
             "e12_code": ROOT / "members/zhenci/lstm/e12/train_e12.py",
             "shared_preprocessing": ROOT / "shared/preprocessing.py",
             "shared_features": ROOT / "shared/features.py"}
    result = {name: sha256(path) for name, path in paths.items()}
    result["e12_code"] = archived_source_hash(paths["e12_code"], result["e12_code"])
    with (E12_OUTPUT / "summary.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    if (summary.get("input_sha256", {}).get("f0") != result["original_f0"]
            or summary.get("architecture", {}).get("window") != 20
            or summary.get("architecture", {}).get("hidden_size") != HIDDEN
            or summary.get("architecture", {}).get("head_dropout") != HEAD_DROPOUT
            or summary.get("training", {}).get("epochs_fixed") != EPOCHS
            or summary.get("training", {}).get("batch_size") != BATCH_SIZE
            or summary.get("training", {}).get("initial_learning_rate") != LEARNING_RATE
            or summary.get("training", {}).get("seed") != SEED):
        raise ValueError("Archived E12 recipe or original F0 changed")
    return result


def archived_controls() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(E12_OUTPUT / "validation_predictions.csv",
                       dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    metrics = pd.read_csv(E12_OUTPUT / "fold_metrics.csv", dtype={"fold": str})
    curves = pd.read_csv(E12_OUTPUT / "learning_curves.csv", dtype={"fold": str})
    validate_keys(rows.loc[rows["fold"].eq("2023"), KEYS])
    if (rows.duplicated(["fold", *KEYS]).any()
            or set(rows["fold"]) != {f.name for f in FOLDS}
            or set(metrics["arm"]) != {"e1", "e3", "e12"}
            or set(curves["epoch"]) != set(range(1, EPOCHS + 1))
            or not rows["e12"].eq(rows["p_lstm"].gt(.5).astype(int)).all()):
        raise ValueError("Archived E12 control rows or learning curves changed")
    return rows, metrics, curves


def load_fold(market: pd.DataFrame, f0: pd.DataFrame, target_ids: list[str], fold,
              archived: pd.DataFrame) -> tuple[dict, dict, list[str], pd.DataFrame, BaselinePreprocessor]:
    train_rows, valid_rows = split_market(market, target_ids, fold)
    preprocessor = BaselinePreprocessor().fit(join_f0(train_rows, f0, include_label=False)[list(F0_FEATURES)])
    train, valid, context_names = _sequence_data(market, f0, fold, target_ids,
                                                  preprocessor, train_rows, valid_rows)
    control = archived.loc[archived["fold"].eq(fold.name)]
    if (len(valid["keys"]) != len(control)
            or set(map(tuple, valid["keys"].to_numpy()))
            != set(map(tuple, control[KEYS].to_numpy()))):
        raise ValueError("E13 evaluation keys differ from archived E12/E3")
    for arm in ("e12", "e3"):
        official = evaluate_predictions(valid_rows, control[KEYS + [arm]].rename(columns={arm: "predicted_y"}))
        original = pd.read_csv(E12_OUTPUT / "fold_metrics.csv", dtype={"fold": str})
        expected = original.loc[original["fold"].eq(fold.name) & original["arm"].eq(arm), "official_score"].iloc[0]
        if not np.isclose(official["official_score"], expected, rtol=0, atol=1e-12):
            raise ValueError(f"Archived {arm} official score failed to reproduce")
    return train, valid, context_names, valid_rows, preprocessor


def run_folder(output: Path, window: int, seed: int, fold: str) -> Path:
    return output / "runs" / f"window_{window}_seed_{seed}_fold_{fold}"


def saved_run(output: Path, window: int, seed: int, fold: str,
              hashes: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    folder = run_folder(output, window, seed, fold)
    if not folder.exists():
        return None
    with (folder / "completed.json").open(encoding="utf-8") as file:
        done = json.load(file)
    if (done.get("window"), done.get("seed"), done.get("fold"), done.get("input_hashes")) != (
            window, seed, fold, hashes):
        raise ValueError(f"Saved E13 run does not match current inputs: {folder}")
    for name, expected in done["file_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Saved E13 run changed: {folder / name}")
    return (pd.read_csv(folder / "fold_metrics.csv", dtype={"fold": str}),
            pd.read_csv(folder / "learning_curves.csv", dtype={"fold": str}),
            pd.read_csv(folder / "validation_predictions.csv", dtype={"stock_id": str, "fold": str},
                        parse_dates=["date"]))


def train_run(output: Path, fold, train: dict, valid: dict, context_names: list[str],
              valid_rows: pd.DataFrame, window: int, seed: int,
              hashes: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing = saved_run(output, window, seed, fold.name, hashes)
    if existing is not None:
        print(f"Reusing completed window={window} seed={seed} fold={fold.name}", flush=True)
        return existing
    folder = run_folder(output, window, seed, fold.name)
    thin_train = slice_window(train, window)
    thin_valid = slice_window(valid, window)
    model, curve, probability, seconds, predict_seconds = fit_candidate(
        thin_train, thin_valid, len(context_names), fold.name, window, seed)
    predicted = thin_valid["keys"].copy()
    predicted.insert(0, "fold", fold.name)
    predicted.insert(1, "window", window)
    predicted.insert(2, "seed", seed)
    predicted["p_up"] = probability.astype(float)
    predicted["predicted_y"] = predicted["p_up"].gt(.5).astype(int)
    official = evaluate_predictions(valid_rows, predicted[KEYS + ["predicted_y"]])
    metric = pd.DataFrame([{**official, "fold": fold.name, "window": window, "seed": seed,
                            "predicted_up_rate": float(predicted["predicted_y"].mean()),
                            "mean_probability": float(predicted["p_up"].mean()),
                            "median_probability": float(predicted["p_up"].median()),
                            "p10_probability": float(predicted["p_up"].quantile(.1)),
                            "p90_probability": float(predicted["p_up"].quantile(.9)),
                            "training_and_curve_seconds": seconds,
                            "final_inference_seconds": predict_seconds,
                            "training_sequences": len(thin_train["keys"])}])
    folder.mkdir(parents=True, exist_ok=False)
    metric.to_csv(folder / "fold_metrics.csv", index=False)
    curve.to_csv(folder / "learning_curves.csv", index=False)
    predicted.to_csv(folder / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), folder / "model.pt")
    files = ("fold_metrics.csv", "learning_curves.csv", "validation_predictions.csv", "model.pt")
    with (folder / "completed.json").open("w", encoding="utf-8") as file:
        json.dump({"window": window, "seed": seed, "fold": fold.name,
                   "input_hashes": hashes,
                   "file_sha256": {name: sha256(folder / name) for name in files}}, file, indent=2)
        file.write("\n")
    print(f"Completed window={window} seed={seed} fold={fold.name}: "
          f"official score {official['official_score']:.8f}", flush=True)
    return metric, curve, predicted


def selected_shorter_window(output: Path, hashes: dict[str, str]) -> tuple[int, dict[int, float]]:
    scores = {}
    for window in (5, 10):
        by_year = {}
        for fold in FOLDS:
            saved = saved_run(output, window, SEED, fold.name, hashes)
            if saved is None:
                raise ValueError("Run the initial 5/10-session comparison first")
            by_year[fold.name] = float(saved[0]["official_score"].iloc[0])
        scores[window] = cv_score(by_year)
    winner = max((5, 10), key=lambda window: (scores[window], -window))
    return winner, scores


def consolidate(output: Path, hashes: dict[str, str], archived: pd.DataFrame,
                archived_metrics: pd.DataFrame, archived_curves: pd.DataFrame,
                winner: int, initial_scores: dict[int, float]) -> Path:
    metrics, curves, predictions = [], [], []
    for fold in FOLDS:
        baseline = archived_metrics.loc[archived_metrics["fold"].eq(fold.name)]
        for arm in ("e3", "e12"):
            row = baseline.loc[baseline["arm"].eq(arm)].iloc[0].to_dict()
            metrics.append({"fold": fold.name, "window": 20 if arm == "e12" else 0,
                            "seed": 42 if arm == "e12" else -1, "arm": arm,
                            "official_score": row["official_score"], "accuracy": row["accuracy"],
                            "n_eval": row["n_eval"],
                            "predicted_up_rate": row["predicted_up_rate"],
                            "mean_probability": row["mean_probability"]})
        original = archived.loc[archived["fold"].eq(fold.name),
                                ["fold", *KEYS, "p_lstm", "e12"]].rename(
                                    columns={"p_lstm": "p_up", "e12": "predicted_y"})
        original.insert(1, "window", 20)
        original.insert(2, "seed", 42)
        predictions.append(original)
        control_curve = archived_curves.loc[archived_curves["fold"].eq(fold.name)].copy()
        control_curve.insert(1, "window", 20)
        control_curve.insert(2, "seed", 42)
        curves.append(control_curve)
        for window in (5, 10, 20):
            seeds = SEEDS if window == winner or window == 20 else (42,)
            for seed in seeds:
                if (window, seed) == (20, 42):
                    continue
                saved = saved_run(output, window, seed, fold.name, hashes)
                if saved is None:
                    raise ValueError(f"Missing E13 run: window={window} seed={seed} fold={fold.name}")
                metric, curve, predicted = saved
                metrics.append({**metric.iloc[0].to_dict(), "arm": f"window_{window}"})
                curves.append(curve)
                predictions.append(predicted)
    metric_table = pd.DataFrame(metrics)
    curve_table = pd.concat(curves, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    outcome = archived[["fold", *KEYS, "realized_return", "actual_up"]]
    paired = prediction_table.merge(outcome, on=["fold", *KEYS], how="left",
                                    validate="many_to_one", indicator=True)
    if not paired["_merge"].eq("both").all() or paired["realized_return"].isna().any():
        raise ValueError("E13 evaluation returns do not match archived E12 keys")
    paired = paired.drop(columns="_merge")
    paired["contribution"] = (2 * paired["predicted_y"] - 1) * paired["realized_return"]
    daily = paired.groupby(["window", "seed", "fold", "date"], as_index=False).agg(
        daily_score=("contribution", "mean"), score_sum=("contribution", "sum"),
        n=("contribution", "size"))
    daily_stats = daily.groupby(["window", "seed", "fold"], as_index=False).agg(
        trading_days=("date", "size"), negative_days=("daily_score", lambda x: int((x < 0).sum())),
        median_daily_score=("daily_score", "median"),
        p10_daily_score=("daily_score", lambda x: float(x.quantile(.1))),
        p90_daily_score=("daily_score", lambda x: float(x.quantile(.9))))
    probability = paired.groupby(["window", "seed", "fold"], as_index=False).agg(
        mean_probability=("p_up", "mean"), median_probability=("p_up", "median"),
        p10_probability=("p_up", lambda x: float(x.quantile(.1))),
        p90_probability=("p_up", lambda x: float(x.quantile(.9))),
        predicted_up_rate=("predicted_y", "mean"), actual_up_rate=("actual_up", "mean"))
    cv_rows = []
    for window in WINDOWS:
        for seed in (SEEDS if window == winner or window == 20 else (42,)):
            rows = metric_table.loc[metric_table["window"].eq(window)
                                    & metric_table["seed"].eq(seed)]
            if len(rows) != 3:
                raise ValueError("Missing fold in seed-level CV summary")
            cv_rows.append({"window": window, "seed": seed,
                            "equal_year_cv_score": cv_score(dict(zip(rows["fold"],
                                                                    rows["official_score"])))})
    cv_table = pd.DataFrame(cv_rows)
    summary = {"experiment": "E13 LSTM sequence-length comparison",
               "control": "archived E12 window 20 seed 42",
               "windows": list(WINDOWS), "robustness_seeds": list(SEEDS),
               "best_shorter_window_by_seed42_cv": winner,
               "initial_seed42_shorter_cv_scores": {str(k): v for k, v in initial_scores.items()},
               "evaluation_keys": {f.name: len(archived.loc[archived["fold"].eq(f.name)])
                                   for f in FOLDS},
               "training_keys": "identical 20-session E12 warm-up for all windows",
               "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE, "optimizer": "Adam",
                            "loss": "BCEWithLogitsLoss", "threshold": .5,
                            "gradient_clip_norm": 1.0, "hidden_size": HIDDEN,
                            "head_dropout": HEAD_DROPOUT},
               "input_sha256": hashes,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "torch": torch.__version__}}
    for name, table in (("fold_metrics", metric_table), ("learning_curves", curve_table),
                        ("validation_predictions", paired), ("daily_scores", daily),
                        ("daily_distribution", daily_stats), ("probability_summary", probability),
                        ("seed_cv", cv_table)):
        table.to_csv(output / f"{name}.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"E13 selected shorter window {winner}; seed-42 means {initial_scores}", flush=True)
    print(cv_table.to_string(index=False), flush=True)
    return output


def run(stage: str = "all", output: Path = OUTPUT) -> Path:
    if stage not in {"initial", "robustness", "all"}:
        raise ValueError("stage must be initial, robustness, or all")
    hashes = input_hashes()
    archived, archived_metrics, archived_curves = archived_controls()
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        with provenance_path.open(encoding="utf-8") as file:
            if json.load(file) != hashes:
                raise ValueError("E13 output directory was created from different inputs")
    else:
        with provenance_path.open("w", encoding="utf-8") as file:
            json.dump(hashes, file, indent=2)
            file.write("\n")
    target_ids, _ = load_metadata(ROOT / "data")
    market = load_market(ROOT / "data")
    market = market.loc[market["date"].le("2025-12-31")].reset_index(drop=True)
    f0 = load_f0_dataset(F0_PATH)
    f0 = f0.loc[f0["date"].le("2025-12-31")].reset_index(drop=True)
    if stage in {"initial", "all"}:
        for fold in FOLDS:
            train, valid, context_names, valid_rows, _ = load_fold(
                market, f0, target_ids, fold, archived)
            for window in (5, 10):
                train_run(output, fold, train, valid, context_names,
                          valid_rows, window, SEED, hashes)
    winner, initial_scores = selected_shorter_window(output, hashes)
    if stage in {"robustness", "all"}:
        for fold in FOLDS:
            train, valid, context_names, valid_rows, _ = load_fold(
                market, f0, target_ids, fold, archived)
            for window in (20, winner):
                for seed in (43, 44):
                    train_run(output, fold, train, valid, context_names,
                              valid_rows, window, seed, hashes)
        return consolidate(output, hashes, archived, archived_metrics,
                           archived_curves, winner, initial_scores)
    print(f"Initial comparison complete: {initial_scores}; best shorter window {winner}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("initial", "robustness", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.stage, args.output_dir))
