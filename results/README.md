# Experiment results

This directory is reserved for future validation and holdout outputs, such as per-fold metrics, prediction rows, and configuration records. No measured model results are present yet. `.gitkeep` preserves the directory.

The prepared [鄭振慈 XGBoost E1 runner](../members/鄭振慈/README.md) will write its three validation-fold metrics, predictions, and configuration summary under `zheng_xgboost_e1_initial_v1/` when run. It has not been run yet.

[`shared/evaluation.py`](../shared/evaluation.py) defines prediction-key validation, the official score, accuracy, and the equal-weight three-fold CV score. It does not train models or automatically write result files. Keep E1 and E2 records identifiable and preserve each fold's score; use the Jan–May 2026 holdout only after configuration selection. Generated files here are gitignored.
