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

## Save a provisional E1 model

[`save_e1.py`](save_e1.py) refits one chosen XGBoost E1 configuration on an explicitly selected training window ending no later than 2025-12-31. It requires the matching validation `summary.json`, a seed from the shared five-seed list, and a new artifact directory. For example, after choosing the full-sample configuration and the pre-holdout training cutoff:

```text
uv run python -m members.zhenci.save_e1 --cv-summary results/zheng_xgboost_e1_five_seeds_v1/summary.json --train-end 2025-12-31 --seed 42 --output-dir results/zhenci_e1_full_pre_holdout_artifact
```

The output contains `model.json`, `preprocessor.json`, snapshots of the F0 feature and preprocessing source, and `manifest.json` with the feature order, threshold, configuration, data fingerprint, and dependency versions. The script checks that reloaded artifacts reproduce a few training predictions. It does not evaluate the holdout or generate a competition submission. The example is a provisional pre-holdout checkpoint; the final deployment-refit window and seed or ensemble policy remain separate decisions.

## Export predictions when the input dates are complete

[`predict_e1.py`](predict_e1.py) loads a saved E1 artifact, applies its frozen preprocessing and model to the required F0 stock/date rows, and writes the exact official template format. It requires every template key to have an F0 row and refuses incomplete inputs:

```text
uv run python -m members.zhenci.predict_e1 --artifact-dir results/zhenci_e1_full_pre_holdout_artifact --template sample_submission.csv --f0-path data/processed/f0.csv --output submissions/zhenci_e1_phase1.csv
```

The supplied Phase 1 template includes 20 September 2026 dates beyond the raw data's August 31 endpoint. Updated, verified source values are required to rebuild F0 before the full file can be exported. The exporter does not obtain market data, infer technical-indicator settings, or upload the CSV.
