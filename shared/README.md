# Shared experiment code

These modules provide the common, model-independent pipeline. All four members use the same E1 feature construction, fold keys, preprocessing, and scoring.

| File | Use |
|---|---|
| [`__init__.py`](__init__.py) | Marks `shared` as a Python package. |
| [`data_loader.py`](data_loader.py) | Loads stock IDs, stock CSVs, and the global CSV; validates source dates, keys, and labels. |
| [`features.py`](features.py) | Builds all 21 logical F0 fields with prior-session stock lags and strict-prior-date global alignment; generates and loads `data/processed/f0.csv`. |
| [`splits.py`](splits.py) | Selects the agreed expanding validation folds, five E1 seeds, and independent holdout from actual rows. |
| [`preprocessing.py`](preprocessing.py) | Defines F0 columns and the training-fold-fitted E1 `BaselinePreprocessor`. |
| [`evaluation.py`](evaluation.py) | Checks complete binary predictions, calculates the official score and accuracy, and summarizes the three yearly scores across five E1 seeds. |
| [`submission.py`](submission.py) | Validates an actual competition template and exports long-form binary predictions in its required wide CSV layout. |

The usual E1 flow is: load source rows, build or load F0, select a fold by stock/date keys, fit preprocessing on its training rows, train a member's model, predict all evaluation keys, then score. Labels and evaluation prices come from the market loader, separate from F0. See the [team guide](../context/README.md) for an example and the [project README](../README.md#data-quality-and-timing-assumptions) for unresolved data timing assumptions.

## What members call in E1

Run these from the repository root so paths such as `data/processed/f0.csv` resolve correctly. The [team guide's worked example](../context/README.md#use-the-shared-f0-file-in-e1) shows the feature join and preprocessing calls together.

| Step | Shared call | Input → output |
|---|---|---|
| 1. Load IDs and source rows | `load_metadata("data")`, `load_market("data")` from `data_loader.py` | The 50 target IDs and 100-stock historical rows. Historical `y` is the training label; adjusted prices are used later for scoring. |
| 2. Load features | `load_f0_dataset("data/processed/f0.csv")` from `features.py` | One F0 row per `(stock_id, date)`, with 21 logical input fields and no label. If the file is absent, run `uv run python -m shared.features --data-dir data --output data/processed/f0.csv` once to rebuild it from source data. |
| 3. Select dates | `split_market(market, target_ids, fold)` with `fold` from `FOLDS` in `splits.py` | Training rows from all 100 stocks and evaluation rows from the 50 targets. Join each side to F0 by both keys. |
| 4. Prepare inputs | A fresh `BaselinePreprocessor` from `preprocessing.py` for each fold | `fit_transform` on that fold's training F0 rows; `transform` on its evaluation F0 rows. `F0_FEATURES` selects the agreed inputs, excluding `date` and `y`. |
| 5. Score predictions | `evaluate_predictions(evaluation_rows, predictions)` from `evaluation.py` | Official score, accuracy, and evaluated row count. `predictions` must contain exactly one binary `predicted_y` for every required `(stock_id, date)` key. |
| 6. Select configuration | `e1_seed_summary({seed: {"2023": score_2023, "2024": score_2024, "2025": score_2025} for seed in E1_SEEDS})` | Returns each seed's equal-weight three-year CV score, their mean (the selection score), population standard deviation, minimum, and maximum. The input must contain actual scores for all five seeds; keep the independent `HOLDOUT` out of this choice. |
| 7. Export when ready | `write_submission(predictions, template_path, output_path, target_ids)` from `submission.py` | A CSV with the exact IDs and date columns in the actual Phase 1 or Phase 2 template. This needs a complete prediction for every template cell. |

`load_global("data")` in `data_loader.py` and `build_f0_dataset` in `features.py` are used when rebuilding F0; a member using the generated E1 file does not need to recalculate global alignment or stock lags. `evaluation_keys(evaluation_rows)` in `splits.py` can extract the required `(stock_id, date)` pairs. `aligned_predictions` in `evaluation.py` can inspect the validated price/prediction join. None of these modules trains a model.
