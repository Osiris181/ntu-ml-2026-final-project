# 鄭振慈

Personal experiment workspace. The chosen primary approach is **XGBoost**.

Experiment files are grouped under [E1](e1/README.md), [E2](e2/README.md), and [E3](e3/README.md). The [E4 folder](e4/README.md) records the proposed next study; it contains no implementation or results yet. Each folder has a file guide or design. The generated `results/` and `submissions/` paths retain their existing names.

[`train_e1.py`](e1/train_e1.py) runs one fixed XGBoost configuration for seeds `42`, `64`, `128`, `256`, and `512` over the three expanding validation folds. It uses `subsample=0.8` and `colsample_bytree=0.8` to sample rows and features during training. It loads the shared [`F0 dataset`](../../data/processed/README.md), fits a fresh [baseline preprocessor](../../shared/preprocessing.py) on each training fold, predicts every target-stock validation row for each seed, and uses the [official shared scorer](../../shared/evaluation.py). It does not use the Jan–May 2026 holdout for selection, train a deployment model, or create a submission.

From the repository root, with `uv` installed:

```text
uv run python -m members.zhenci.e1.train_e1
```

You can also run `uv run python members/zhenci/e1/train_e1.py` from the repository root. The script resolves the shared package and its default data and results paths from the repository location, so direct file execution works too.

`uv run` installs the dependencies declared in `pyproject.toml` and creates a lockfile if needed. The script writes `fold_metrics.csv` (15 seed/fold rows), `validation_predictions.csv` (each prediction tagged with its seed), and `summary.json` (per-seed CV, five-seed mean and standard deviation, configuration, threshold, and package versions) under `results/zheng_xgboost_e1_sampling_v1/`. Those generated files are gitignored. The script refuses to overwrite an existing output directory; pass a different `--output-dir` for another run. See the [team guide](../../context/README.md) for the shared E1 inputs and outputs.

Sampling is a model-specific choice recorded as `xgboost_sampling_v1`; the shared F0 features, folds, preprocessing, seed list, threshold, and official scorer are unchanged. Different seeds may now yield different probabilities, but distinct official scores are not guaranteed.

## E2: tune XGBoost settings

[`train_e2.py`](e2/train_e2.py) compares a **prespecified four-configuration grid**. It varies tree depth (`3` or `5`) and learning-rate/round pairs (`0.05` with 300 rounds or `0.10` with 200 rounds), with fixed sampling and regularization. This first E2 run keeps the original `data/processed/f0.csv`, the same 21 F0 features, shared baseline preprocessing, `0.5` prediction threshold, and the same three folds, five seeds, target keys, and official scorer as E1. It does not read the submission score, updated post-August data, or Jan–May 2026 holdout.

From the repository root:

```text
uv run python -m members.zhenci.e2.train_e2
```

The script writes `fold_metrics.csv` (60 configuration/seed/fold rows), `validation_predictions.csv`, and `summary.json` under `results/zhenci_xgboost_e2_grid_v1/`. The summary records all settings, compute budget, each seed's three yearly scores, the five-seed mean and standard deviation for each configuration, and the configuration selected by the highest five-seed mean. It refuses to overwrite an existing output directory. This is validation only: it does not refit or save an E2 model, evaluate the independent holdout, or create a submission. New features or preprocessing can be tested later as separate E2 changes.

### E2 feature experiment

[`features_e2.py`](e2/features_e2.py) adds three member-owned F1 fields to the unchanged shared F0 file: `prior_overnight_gap` (`open[t-1] / close[t-2] - 1`), `momentum_10d` (`close[t-1] / close[t-11] - 1`), and `close_volatility_5d` (standard deviation of the five close-to-close returns ending at `t-1`). Its F1 preprocessor applies the ordinary shared F0 transforms and fits median imputation and standard scaling for the three new numeric fields on each training fold only. The feature builder uses supplied historical stock observations; actual publication timing remains the central unresolved assumption.

[`train_e2_features.py`](e2/train_e2_features.py) reads the configuration selected by the completed E2 grid, then runs a paired F0 control versus F1 test using that same model configuration, threshold, three folds, five seeds, target keys, and official scorer. It reads the original `data/processed/f0.csv`, not later submission outcomes. From the repository root:

```text
uv run python -m members.zhenci.e2.train_e2_features
```

Results go to `results/zhenci_xgboost_e2_features_v1/`: 30 fold/seed/feature-set metric rows, validation predictions, and a summary showing both five-seed means and `F1 - F0`. The script refuses to overwrite existing results and does not train a deployment artifact, evaluate the independent holdout, or create a submission.

### E2 training-weight experiment

[`train_e2_weighted.py`](e2/train_e2_weighted.py) tests one separate change to the selected full-sample E1 XGBoost configuration: give each training row a weight proportional to the **absolute adjusted intraday return**, normalized to mean weight one within that fold. The official score gives larger moves more impact, while ordinary binary log loss treats each row equally. Flat training sessions get zero weight. The script checks that historical `y` agrees with adjusted price direction. It uses training-row prices only for weights; validation realized prices remain exclusively in the shared scorer. F0, the baseline preprocessor, all three folds, five seeds, target keys, and the `0.5` threshold are unchanged.

From the repository root:

```text
uv run python -m members.zhenci.e2.train_e2_weighted
```

The output directory `results/zhenci_xgboost_e2_weighted_v1/` contains an unweighted control and weighted arm, each evaluated on all 15 seed/fold combinations, with metrics, predictions, and a summary of their five-seed CV score difference. This is an experiment based on the published scoring rule; it does not use the Phase 1 submission result or the Jan–May 2026 holdout for selection, and it does not save or submit a model.

### E2 return-regression experiment

[`train_e2_return_regression.py`](e2/train_e2_return_regression.py) compares the selected E1 direction classifier with an XGBoost model trained to predict the **adjusted intraday return** directly. A predicted return above zero becomes `1`; zero or below becomes `0`. The regression target comes only from each fold's training rows. This changes the XGBoost learning objective to `reg:squarederror` while holding the tree settings, F0, baseline preprocessing, folds, seeds, target keys, and official scorer fixed. The question is whether a model aimed at expected return beats a direction-only classifier on the agreed validation score; a better submission score is not guaranteed.

From the repository root:

```text
uv run python -m members.zhenci.e2.train_e2_return_regression
```

It writes 30 control/regression seed/fold metric rows, validation predictions, and both five-seed scores to `results/zhenci_xgboost_e2_return_regression_v1/`. It does not use the Phase 1 submission outcome or independent holdout for model selection and does not save or submit a deployment model.

The E1 reference has `subsample=1.0` and `colsample_bytree=1.0`, so its five seeds can give identical models. To test seed sensitivity, use a separate sampled mode:

```text
uv run python -m members.zhenci.e2.train_e2_return_regression --sampled
```

This changes **both** arms to `subsample=0.8` and `colsample_bytree=0.8`, records the settings in the summary, and writes to `results/zhenci_xgboost_e2_return_regression_sampled_v2/`. Compare the classifier and regressor within that run. Its control also differs from the full-sample E1 configuration, so compare the sampled winner with E1's score separately. Different seeds can produce different outputs, but distinct official scores are not guaranteed.

## E3: Optuna search on F0

[`train_e3.py`](e3/train_e3.py) is the third XGBoost experiment. It uses Optuna to search tree depth, learning rate, boosting rounds, minimum child weight, row/column sampling, and L2 regularization. The first two trials reproduce the full-sample and sampled E1 settings as direct controls. The default is **20 total trials**, including those two controls. Each trial trains all five required seeds on each of the same three expanding folds and is ranked by the shared, equally weighted official CV score. The Optuna sampler uses seed `42` and runs sequentially; the model seeds remain `42`, `64`, `128`, `256`, and `512`.

This experiment keeps the 21 F0 features, fold-fitted baseline preprocessing, `0.5` threshold, training/evaluation stock universes, and official scorer. It does not use the independent holdout, updated submission data, or submission feedback. Its result rows use `experiment=E2`, the shared schema's name for member optimization; **E3** is the local experiment label.

From the repository root:

```text
uv run python -m members.zhenci.e3.train_e3
```

Use `--trials N` to change the total trial budget (minimum `2`) or `--output-dir PATH` for another run. `uv run` installs Optuna from `pyproject.toml` if needed. The default output is `results/zhenci_xgboost_e3_optuna_v1/`. It contains `summary.json` with every trial's settings and all 15 fold/seed scores, `fold_metrics.csv` for every fit, and `selected_validation_predictions.csv` for the selected trial. `selected_feature_importance.csv` ranks encoded columns; `selected_logical_feature_importance.csv` combines one-hot columns back into the 21 F0 fields. Importance is mean normalized XGBoost split gain over the selected trial's 15 fitted validation models. It is descriptive and does not determine feature removal in this run. The script refuses to overwrite an existing output directory and does not save or submit a model.

## E3 alternative: change XGBoost's training data

[`train_e3_regime.py`](e3/train_e3_regime.py) tests a different XGBoost approach that can still use the same F0 features at submission time. It compares six fixed policies: train on all 100 stocks or just the 50 target stocks, each with ordinary training, a 730-calendar-day weight half-life, or a 365-calendar-day weight half-life. A half-life gives observations that many days older half the training weight. It uses only each fold's training dates to set weights; the validation target rows and official scorer stay the same. The all-100 ordinary policy is the full-sample E1 control. All candidates use the same XGBoost tree settings, F0, fold-fitted baseline preprocessing, `0.5` threshold, three folds, and five seeds. Because this fixed XGBoost configuration has no row or column sampling, identical results across seeds are possible and expected.

Run from the repository root:

```text
uv run python -m members.zhenci.e3.train_e3_regime
```

The default `results/zhenci_xgboost_e3_regime_v1/` directory receives `fold_metrics.csv` for all 90 fits, `summary.json` with all six configurations and the selected CV winner, and `selected_validation_predictions.csv`. The selected alternative must beat the E1 control before it can be saved as E3. The code never uses Phase 1 submission feedback, updated post-August labels, or the Jan–May 2026 holdout for selection. The result rows use `experiment=E2` because that is the shared schema's category for member optimization; **E3** is this member's experiment label.

If an alternative wins, [`save_e3_regime.py`](e3/save_e3_regime.py) refits that policy on the training side of the shared holdout split (through 2025), using seed `42`, and freezes its model and F0 preprocessor in `results/zhenci_e3_regime_pre_holdout_artifact/`. It refuses to save when E1 remains best:

```text
uv run python -m members.zhenci.e3.save_e3_regime
```

With complete updated F0 inputs for every official template key, [`predict_e3_regime.py`](e3/predict_e3_regime.py) reuses the checked frozen-artifact exporter and writes `submissions/zhenci_e3_regime_phase1.csv` in official template order:

```text
uv run python -m members.zhenci.e3.predict_e3_regime
```

These are provisional pre-holdout scripts; no E3 result or artifact exists until the commands are run. Feature publication timing and supplied technical-indicator settings retain the central data-quality limitations. A CV win does not guarantee a higher Phase 1 submission score.

## E3 follow-up: relative features, thresholds, and DART

[`features_e3_relative.py`](e3/features_e3_relative.py) builds three member-owned fields from the already lagged F0 values on each prediction date: previous-session intraday return minus the cross-stock median, five-session momentum minus the cross-stock median, and previous relative-volume percentile among available stocks. It preserves every F0 stock/date row and fits extra imputation/scaling on each training fold only. No target-day prices or labels enter these calculations. Peer aggregates use all stock rows available for that date, so future submission inference needs the corresponding updated stock-history inputs, not just the 50 output IDs.

[`train_e3_feature_sets.py`](e3/train_e3_feature_sets.py) compares four fixed feature sets with the full-sample E1 XGBoost settings: F0 control, F0 without the four global fields, F0 plus the three relative fields, and relative fields without globals. Every variant uses the same three folds, five seeds, target evaluation keys, `0.5` cutoff, and official score. Run:

```text
uv run python -m members.zhenci.e3.train_e3_feature_sets
```

Results go to `results/zhenci_xgboost_e3_feature_sets_v1/`: all 60 fit metrics, a summary of every variant's CV score, and the selected variant's validation probabilities. Removing globals or adding relative features is an E3 choice; the shared E1 F0 definition is unchanged.

[`evaluate_e3_threshold.py`](e3/evaluate_e3_threshold.py) reads saved validation probabilities and compares fixed cutoffs from `0.35` to `0.65`, using the official score. It reports both the threshold chosen from all three CV years and a **walk-forward** check: 2023 uses `0.5`, 2024 chooses from 2023, and 2025 chooses from 2023–2024. The same-period chosen score is tuned on those folds and should not be treated as independent evidence. By default it reads the completed full-sample E1 probabilities; use `--predictions PATH --output-dir PATH` to inspect a different completed single-configuration run. It does not fit a model:

```text
uv run python -m members.zhenci.e3.evaluate_e3_threshold
```

Results go to `results/zhenci_xgboost_e3_threshold_v1/`. A promising threshold would need to be frozen in its own validated artifact before submission; the current E1 exporter intentionally accepts only its saved `0.5` cutoff.

[`train_e3_dart.py`](e3/train_e3_dart.py) compares the original XGBoost `gbtree` booster with DART at dropout rates `0.05` and `0.10`, holding F0, preprocessing, rounds, folds, seeds, target keys, and scorer fixed. Validation prediction explicitly uses all trained rounds with `iteration_range=(0, 100)`; a future DART exporter must use the same rule. DART may take longer than the other runs. Run:

```text
uv run python -m members.zhenci.e3.train_e3_dart
```

Results go to `results/zhenci_xgboost_e3_dart_v1/`: all 45 fit metrics, all three configurations' CV summaries, and selected validation probabilities. These three follow-up scripts are **validation studies only**. They do not refit, save, or submit a new model. If a specific configuration wins, its exact feature/preprocessing/threshold/booster settings must be carried into a new frozen artifact before a submission is generated. No submitted score is used to choose among them.

### E3 market-regime threshold study

[`evaluate_e3_market_regime.py`](e3/evaluate_e3_market_regime.py) tests whether the prior eligible TAIEX return in the shared F0 file helps choose a binary prediction threshold. It reads saved E1 validation probabilities, so it does **not** retrain XGBoost. The six fixed candidate rules include the E1 `0.5` control, two unconditional threshold controls, and three rules that become more conservative about predicting up when the prior TAIEX return is negative. The regime rules use `0.5` for missing or stale TAIEX values; unconditional controls always use their fixed threshold. The study uses the same three target-stock validation folds, all five E1 seeds, and the official score. It reports both same-period CV selection and a walk-forward check that chooses each year's rule from earlier validation years only.

```text
uv run python -m members.zhenci.e3.evaluate_e3_market_regime
```

Results go to `results/zhenci_xgboost_e3_market_regime_v2/`. The earlier `v1` fixed-threshold controls incorrectly used `0.5` on missing-TAIEX dates; `v2` corrects that comparison. This is a validation study only; same-period selection is optimistic. It does not read the independent 2026 holdout or submission score.

For a provisional Phase 1 submission file, [`predict_e3_market_regime.py`](e3/predict_e3_market_regime.py) uses the saved full-sample E1 model and preprocessor, reads the selected rule from the corrected `v2` summary, applies it to updated F0 values, and writes the official template without retraining:

```text
uv run python -m members.zhenci.e3.predict_e3_market_regime
```

The default output is `submissions/zhenci_e3_taiex_phase1.csv`. It requires the saved E1 artifact, `data/processed/f0_with_updates.csv`, and complete template stock/date rows. The exporter checks artifact hashes and the rule's model configuration before writing. Generating this CSV does not upload it or establish a Phase 2 deployment artifact.

## E5: Test whether removing F0 inputs helps

[`e5/README.md`](e5/README.md) explains the prepared feature-ablation study. [`train_e5_ablation.py`](e5/train_e5_ablation.py) compares the full F0 control with seven prespecified feature removals, applying both the normal `0.5` threshold and the existing fixed E3 TAIEX rule to each model. It keeps the same folds, five seeds, shared preprocessing, and official score. It also records training-versus-validation diagnostics and logical feature importance. Run from the repository root without flags:

```text
uv run python -m members.zhenci.e5.train_e5_ablation
```

The default output is `results/zhenci_xgboost_e5_f0_ablation_v1/`. This is validation only; the runner does not use the independent holdout or Phase 1 submission outcomes, train a deployment model, or generate a submission. A feature is kept or removed according to its validation score, not importance rank alone.

## E6: Test chronological early stopping

[`e6/README.md`](e6/README.md) explains the E6 study. [`train_e6.py`](e6/train_e6.py) uses an earlier year within each training window to choose a boosting-round count, refits on the outer fold's full training data, and compares two fixed XGBoost settings with the saved E3 control. It retains shared F0 and the fixed E3 TAIEX decision rule. Run from the repository root:

```text
uv run python -m members.zhenci.e6.train_e6
```

The default output is `results/zhenci_xgboost_e6_nested_stop_v1/`. E6 is validation only; no deployment model or submission is produced.

## E7: Test score-aligned historical payoff cutoffs

[`e7/README.md`](e7/README.md) explains the payoff-based decision study. [`evaluate_e7.py`](e7/evaluate_e7.py) keeps the existing E1 XGBoost probabilities, fits cutoffs from each fold's earlier historical returns, and compares pooled, prior-TAIEX-regime, and stock-level payoff policies with the fixed E3 control. It does not train a new XGBoost model or use 2026 outcomes. Run from the repository root:

```text
uv run python -m members.zhenci.e7.evaluate_e7
```

The default output is `results/zhenci_xgboost_e7_payoff_v1/`. Any CV advantage remains exploratory until checked on data outside the selection folds; no deployment model or submission is produced.

The completed run selected the stock-level payoff rule. [`evaluate_holdout.py`](e7/evaluate_holdout.py) performs its one-time Jan–May 2026 check using the already saved 2020–2025 E1 model and original F0. It does not refit the model or choose a different rule. See the E7 README for the result and run command.

[`predict_e7.py`](e7/predict_e7.py) exports an exploratory Phase 1 CSV using that fixed E7 rule and the official template. It uses the saved E1 model and updated F0 input, and does not upload the file. The E7 README has its default command and output path.

## Save a provisional E1 model

[`save_e1.py`](e1/save_e1.py) refits one chosen XGBoost E1 configuration on the **training side of the shared `HOLDOUT` split** (2020–2025). It takes the cutoff and reference seed `42` from [`shared/splits.py`](../../shared/splits.py). By default, it reads the existing full-sample E1 `summary.json` and writes to a new `results/zhenci_e1_full_pre_holdout_artifact/` directory:

```text
uv run python -m members.zhenci.e1.save_e1
```

The default summary is fixed to the full-sample run because it had the higher five-seed mean CV score of the two completed E1 configurations; the script does not automatically search for a future best configuration. Pass `--cv-summary` and a new `--output-dir` if a later CV comparison selects something else. The script refuses to overwrite an existing artifact directory.

The output contains `model.json`, `preprocessor.json`, snapshots of the F0 feature and preprocessing source, and `manifest.json` with the feature order, threshold, configuration, data fingerprint, and dependency versions. The script checks that reloaded artifacts reproduce a few training predictions. It does not evaluate the holdout or generate a competition submission. This is a provisional pre-holdout checkpoint; the final deployment-refit window and seed or ensemble policy remain separate decisions.

## Export predictions when the input dates are complete

[`predict_e1.py`](e1/predict_e1.py) loads the saved E1 artifact, applies its frozen preprocessing and model to the required F0 stock/date rows, and writes the exact official template format. By default it uses the artifact directory created by `save_e1.py`, `data/processed/f0_with_updates.csv`, the root `sample_submission.csv`, and the new output path `submissions/zhenci_e1_phase1.csv`. First build the updated F0 file, then run inference:

```text
uv run python -m shared.updated_features
uv run python -m members.zhenci.e1.predict_e1
```

It requires every template key to have an F0 row and refuses incomplete inputs or an existing output file. Override `--artifact-dir`, `--f0-path`, `--template`, or `--output` when using different inputs. The feature builder ignores updated labels, including the October 23 `-1` placeholders. The exporter does not obtain market data, infer technical-indicator settings, or upload the CSV.
