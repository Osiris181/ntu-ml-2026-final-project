# Shared experiment code

These modules provide the common, model-independent pipeline. All four members use the same E1 feature construction, fold keys, preprocessing, and scoring.

| File | Use |
|---|---|
| [`__init__.py`](__init__.py) | Marks `shared` as a Python package. |
| [`data_loader.py`](data_loader.py) | Loads stock IDs, stock CSVs, and the global CSV; validates source dates, keys, and labels. |
| [`features.py`](features.py) | Builds all 21 logical F0 fields with prior-session stock lags and strict-prior-date global alignment; generates and loads `data/processed/f0.csv`. |
| [`splits.py`](splits.py) | Selects the agreed expanding validation folds and independent holdout from actual rows. |
| [`preprocessing.py`](preprocessing.py) | Defines F0 columns and the training-fold-fitted E1 `BaselinePreprocessor`. |
| [`evaluation.py`](evaluation.py) | Checks complete binary predictions, calculates the official score and accuracy, and averages the three yearly validation scores. |
| [`submission.py`](submission.py) | Validates an actual competition template and exports long-form binary predictions in its required wide CSV layout. |

The usual E1 flow is: load source rows, build or load F0, select a fold by stock/date keys, fit preprocessing on its training rows, train a member's model, predict all evaluation keys, then score. Labels and evaluation prices come from the market loader, separate from F0. See the [team guide](../context/README.md) for an example and the [project README](../README.md#data-quality-and-timing-assumptions) for unresolved data timing assumptions.
