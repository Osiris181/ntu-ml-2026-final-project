# Submission outputs

This directory is for generated competition prediction CSVs. No trained-model submission file is present yet; `.gitkeep` preserves the directory. Generated files are gitignored.

[`shared/submission.py`](../shared/submission.py) accepts one long-form `(stock_id, date, predicted_y)` row per required prediction and writes the exact `id` row order and date headers from the supplied template. The current [`sample_submission.csv`](../sample_submission.csv) is the Phase 1 template. Use the actual Phase 2 session template when it becomes available; the exporter cannot infer future dates or obtain market data. It rejects missing, extra, duplicate, or nonbinary predictions.
