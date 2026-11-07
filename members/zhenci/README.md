# 鄭振慈

Personal experiment workspace. The chosen primary approach is **XGBoost**.

[`train_e1.py`](train_e1.py) runs one fixed XGBoost configuration for seeds `42`, `64`, `128`, `256`, and `512` over the three expanding validation folds. It uses `subsample=0.8` and `colsample_bytree=0.8` to sample rows and features during training. It loads the shared [`F0 dataset`](../../data/processed/README.md), fits a fresh [baseline preprocessor](../../shared/preprocessing.py) on each training fold, predicts every target-stock validation row for each seed, and uses the [official shared scorer](../../shared/evaluation.py). It does not use the Jan–May 2026 holdout for selection, train a deployment model, or create a submission.

From the repository root, with `uv` installed:

```text
uv run python -m members.zhenci.train_e1
```

You can also run `uv run python members/zhenci/train_e1.py` from the repository root. The script resolves the shared package and its default data and results paths from the repository location, so direct file execution works too.

`uv run` installs the dependencies declared in `pyproject.toml` and creates a lockfile if needed. The script writes `fold_metrics.csv` (15 seed/fold rows), `validation_predictions.csv` (each prediction tagged with its seed), and `summary.json` (per-seed CV, five-seed mean and standard deviation, configuration, threshold, and package versions) under `results/zheng_xgboost_e1_sampling_v1/`. Those generated files are gitignored. The script refuses to overwrite an existing output directory; pass a different `--output-dir` for another run. See the [team guide](../../context/README.md) for the shared E1 inputs and outputs.

Sampling is a model-specific choice recorded as `xgboost_sampling_v1`; the shared F0 features, folds, preprocessing, seed list, threshold, and official scorer are unchanged. Different seeds may now yield different probabilities, but distinct official scores are not guaranteed.

## E2: tune XGBoost settings

[`train_e2.py`](train_e2.py) compares a **prespecified four-configuration grid**. It varies tree depth (`3` or `5`) and learning-rate/round pairs (`0.05` with 300 rounds or `0.10` with 200 rounds), with fixed sampling and regularization. This first E2 run keeps the original `data/processed/f0.csv`, the same 21 F0 features, shared baseline preprocessing, `0.5` prediction threshold, and the same three folds, five seeds, target keys, and official scorer as E1. It does not read the submission score, updated post-August data, or Jan–May 2026 holdout.

From the repository root:

```text
uv run python -m members.zhenci.train_e2
```

The script writes `fold_metrics.csv` (60 configuration/seed/fold rows), `validation_predictions.csv`, and `summary.json` under `results/zhenci_xgboost_e2_grid_v1/`. The summary records all settings, compute budget, each seed's three yearly scores, the five-seed mean and standard deviation for each configuration, and the configuration selected by the highest five-seed mean. It refuses to overwrite an existing output directory. This is validation only: it does not refit or save an E2 model, evaluate the independent holdout, or create a submission. New features or preprocessing can be tested later as separate E2 changes.

### E2 feature experiment

[`features_e2.py`](features_e2.py) adds three member-owned F1 fields to the unchanged shared F0 file: `prior_overnight_gap` (`open[t-1] / close[t-2] - 1`), `momentum_10d` (`close[t-1] / close[t-11] - 1`), and `close_volatility_5d` (standard deviation of the five close-to-close returns ending at `t-1`). Its F1 preprocessor applies the ordinary shared F0 transforms and fits median imputation and standard scaling for the three new numeric fields on each training fold only. The feature builder uses supplied historical stock observations; actual publication timing remains the central unresolved assumption.

[`train_e2_features.py`](train_e2_features.py) reads the configuration selected by the completed E2 grid, then runs a paired F0 control versus F1 test using that same model configuration, threshold, three folds, five seeds, target keys, and official scorer. It reads the original `data/processed/f0.csv`, not later submission outcomes. From the repository root:

```text
uv run python -m members.zhenci.train_e2_features
```

Results go to `results/zhenci_xgboost_e2_features_v1/`: 30 fold/seed/feature-set metric rows, validation predictions, and a summary showing both five-seed means and `F1 - F0`. The script refuses to overwrite existing results and does not train a deployment artifact, evaluate the independent holdout, or create a submission.

### E2 training-weight experiment

[`train_e2_weighted.py`](train_e2_weighted.py) tests one separate change to the selected full-sample E1 XGBoost configuration: give each training row a weight proportional to the **absolute adjusted intraday return**, normalized to mean weight one within that fold. The official score gives larger moves more impact, while ordinary binary log loss treats each row equally. Flat training sessions get zero weight. The script checks that historical `y` agrees with adjusted price direction. It uses training-row prices only for weights; validation realized prices remain exclusively in the shared scorer. F0, the baseline preprocessor, all three folds, five seeds, target keys, and the `0.5` threshold are unchanged.

From the repository root:

```text
uv run python -m members.zhenci.train_e2_weighted
```

The output directory `results/zhenci_xgboost_e2_weighted_v1/` contains an unweighted control and weighted arm, each evaluated on all 15 seed/fold combinations, with metrics, predictions, and a summary of their five-seed CV score difference. This is an experiment based on the published scoring rule; it does not use the Phase 1 submission result or the Jan–May 2026 holdout for selection, and it does not save or submit a model.

### E2 return-regression experiment

[`train_e2_return_regression.py`](train_e2_return_regression.py) compares the selected E1 direction classifier with an XGBoost model trained to predict the **adjusted intraday return** directly. A predicted return above zero becomes `1`; zero or below becomes `0`. The regression target comes only from each fold's training rows. This changes the XGBoost learning objective to `reg:squarederror` while holding the tree settings, F0, baseline preprocessing, folds, seeds, target keys, and official scorer fixed. The question is whether a model aimed at expected return beats a direction-only classifier on the agreed validation score; a better submission score is not guaranteed.

From the repository root:

```text
uv run python -m members.zhenci.train_e2_return_regression
```

It writes 30 control/regression seed/fold metric rows, validation predictions, and both five-seed scores to `results/zhenci_xgboost_e2_return_regression_v1/`. It does not use the Phase 1 submission outcome or independent holdout for model selection and does not save or submit a deployment model.

The E1 reference has `subsample=1.0` and `colsample_bytree=1.0`, so its five seeds can give identical models. To test seed sensitivity, use a separate sampled mode:

```text
uv run python -m members.zhenci.train_e2_return_regression --sampled
```

This changes **both** arms to `subsample=0.8` and `colsample_bytree=0.8`, records the settings in the summary, and writes to `results/zhenci_xgboost_e2_return_regression_sampled_v2/`. Compare the classifier and regressor within that run. Its control also differs from the full-sample E1 configuration, so compare the sampled winner with E1's score separately. Different seeds can produce different outputs, but distinct official scores are not guaranteed.

## E3: Optuna search on F0

[`train_e3.py`](train_e3.py) is the third XGBoost experiment. It uses Optuna to search tree depth, learning rate, boosting rounds, minimum child weight, row/column sampling, and L2 regularization. The first two trials reproduce the full-sample and sampled E1 settings as direct controls. The default is **20 total trials**, including those two controls. Each trial trains all five required seeds on each of the same three expanding folds and is ranked by the shared, equally weighted official CV score. The Optuna sampler uses seed `42` and runs sequentially; the model seeds remain `42`, `64`, `128`, `256`, and `512`.

This experiment keeps the 21 F0 features, fold-fitted baseline preprocessing, `0.5` threshold, training/evaluation stock universes, and official scorer. It does not use the independent holdout, updated submission data, or submission feedback. Its result rows use `experiment=E2`, the shared schema's name for member optimization; **E3** is the local experiment label.

From the repository root:

```text
uv run python -m members.zhenci.train_e3
```

Use `--trials N` to change the total trial budget (minimum `2`) or `--output-dir PATH` for another run. `uv run` installs Optuna from `pyproject.toml` if needed. The default output is `results/zhenci_xgboost_e3_optuna_v1/`. It contains `summary.json` with every trial's settings and all 15 fold/seed scores, `fold_metrics.csv` for every fit, and `selected_validation_predictions.csv` for the selected trial. `selected_feature_importance.csv` ranks encoded columns; `selected_logical_feature_importance.csv` combines one-hot columns back into the 21 F0 fields. Importance is mean normalized XGBoost split gain over the selected trial's 15 fitted validation models. It is descriptive and does not determine feature removal in this run. The script refuses to overwrite an existing output directory and does not save or submit a model.

## Save a provisional E1 model

[`save_e1.py`](save_e1.py) refits one chosen XGBoost E1 configuration on the **training side of the shared `HOLDOUT` split** (2020–2025). It takes the cutoff and reference seed `42` from [`shared/splits.py`](../../shared/splits.py). By default, it reads the existing full-sample E1 `summary.json` and writes to a new `results/zhenci_e1_full_pre_holdout_artifact/` directory:

```text
uv run python -m members.zhenci.save_e1
```

The default summary is fixed to the full-sample run because it had the higher five-seed mean CV score of the two completed E1 configurations; the script does not automatically search for a future best configuration. Pass `--cv-summary` and a new `--output-dir` if a later CV comparison selects something else. The script refuses to overwrite an existing artifact directory.

The output contains `model.json`, `preprocessor.json`, snapshots of the F0 feature and preprocessing source, and `manifest.json` with the feature order, threshold, configuration, data fingerprint, and dependency versions. The script checks that reloaded artifacts reproduce a few training predictions. It does not evaluate the holdout or generate a competition submission. This is a provisional pre-holdout checkpoint; the final deployment-refit window and seed or ensemble policy remain separate decisions.

## Export predictions when the input dates are complete

[`predict_e1.py`](predict_e1.py) loads the saved E1 artifact, applies its frozen preprocessing and model to the required F0 stock/date rows, and writes the exact official template format. By default it uses the artifact directory created by `save_e1.py`, `data/processed/f0_with_updates.csv`, the root `sample_submission.csv`, and the new output path `submissions/zhenci_e1_phase1.csv`. First build the updated F0 file, then run inference:

```text
uv run python -m shared.updated_features
uv run python -m members.zhenci.predict_e1
```

It requires every template key to have an F0 row and refuses incomplete inputs or an existing output file. Override `--artifact-dir`, `--f0-path`, `--template`, or `--output` when using different inputs. The feature builder ignores updated labels, including the October 23 `-1` placeholders. The exporter does not obtain market data, infer technical-indicator settings, or upload the CSV.
