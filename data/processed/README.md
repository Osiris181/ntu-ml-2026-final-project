# Processed features

`f0.csv` is the generated shared E1 feature dataset. It has one row per supplied stock/date observation, with `date` as a key and the same 21 logical F0 fields for all four approaches. It has no historical label `y` or target-day market values. Missing historical or stale global values remain missing for the fold-fitted preprocessor.

From the repository root, rebuild it with:

```text
python -m shared.features --data-dir data --output data/processed/f0.csv
```

Load it with `shared.features.load_f0_dataset("data/processed/f0.csv")`. Select each fold's rows by `(stock_id, date)` before fitting [`BaselinePreprocessor`](../../shared/preprocessing.py) on that fold's training rows. Obtain labels and evaluation prices separately from [`data/market/`](../market/README.md).

`f0.csv` is gitignored and can be regenerated; do not manually edit it. `.gitkeep` preserves this directory when the generated CSV is absent. The [project README](../../README.md#shared-f0-baseline) defines every feature.
