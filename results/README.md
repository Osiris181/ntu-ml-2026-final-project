# Experiment results

This directory is reserved for future validation and holdout outputs, such as per-fold metrics, prediction rows, and configuration records. No measured model results are present yet. `.gitkeep` preserves the directory.

[`shared/evaluation.py`](../shared/evaluation.py) defines prediction-key validation, the official score, accuracy, and the equal-weight three-fold CV score. It does not train models or automatically write result files. Keep E1 and E2 records identifiable and preserve each fold's score; use the Jan–May 2026 holdout only after configuration selection. Generated files here are gitignored.
