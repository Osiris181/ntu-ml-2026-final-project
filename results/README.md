# Experiment results

This directory holds generated validation and future holdout outputs, such as per-fold metrics, prediction rows, and configuration records. `.gitkeep` preserves the directory when outputs are absent.
Provisional model artifacts may also be saved here in separate directories with `model.json`, `preprocessor.json`, source snapshots, and a manifest. Keep each artifact linked to its selected CV configuration and training cutoff.

The [XGBoost E1 runner](../members/zhenci/README.md) writes `fold_metrics.csv`, `validation_predictions.csv`, and `summary.json` to a separate result directory. Generated files here are gitignored.

[`shared/evaluation.py`](../shared/evaluation.py) defines prediction-key validation, the official score, accuracy, each seed's equal-weight three-fold CV score, and the five-seed selection summary. It does not train models or automatically write result files. Keep E1 and E2 records identifiable and preserve each seed/fold score; use the Jan–May 2026 holdout only after configuration selection.
