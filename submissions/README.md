# Submission outputs

This directory is for generated competition prediction CSVs. No trained-model submission file is present yet; `.gitkeep` preserves the directory. Generated files are gitignored.

[`shared/submission.py`](../shared/submission.py) accepts one long-form `(stock_id, date, predicted_y)` row per required prediction and writes the exact `id` row order and date headers from the supplied template. The current [`sample_submission.csv`](../sample_submission.csv) is the Phase 1 template. Use the actual Phase 2 session template when it becomes available; the exporter cannot infer future dates or obtain market data. It rejects missing, extra, duplicate, or nonbinary predictions.

The [XGBoost E1 exporter](../members/zhenci/README.md#export-predictions-when-the-input-dates-are-complete) uses a saved model artifact and this shared template validator. It requires an F0 row for every template stock/date key before writing a file.

Each member's first submission candidate should come from their CV-selected, refitted **E1 baseline**; E2 comparison comes later. Before any upload, verify the complete input coverage and exact official template, keep a record of the producing configuration and model artifact, and follow the competition's actual account and submission rules. Do not use submission feedback to tune E2 or replace the reserved holdout evaluation.
