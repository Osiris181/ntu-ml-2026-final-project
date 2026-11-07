# Processed features

`f0.csv` is the generated shared E1 feature dataset. It has one row per supplied stock/date observation, with `date` as a key and the same 21 logical F0 fields for all four approaches. It has no historical label `y` or target-day market values. Missing historical or stale global values remain missing for the fold-fitted preprocessor.

From the repository root, rebuild it with:

```text
uv run python -m shared.features --data-dir data --output data/processed/f0.csv
```

Load it with `shared.features.load_f0_dataset("data/processed/f0.csv")`. Select each fold's rows by `(stock_id, date)` before fitting [`BaselinePreprocessor`](../../shared/preprocessing.py) on that fold's training rows. Obtain labels and evaluation prices separately from [`data/market/`](../market/README.md).

The [team guide's E1 example](../../context/README.md#use-the-shared-f0-file-in-e1) shows the exact load, join, `fit_transform`, and `transform` calls. Each fold needs a fresh preprocessor. Missing numeric cells in this CSV are filled using **that fold's training medians**, not by editing or filling the shared CSV in advance.

`f0.csv` is gitignored and can be regenerated; do not manually edit it. `.gitkeep` preserves this directory when the generated CSV is absent. The [project README](../../README.md#shared-f0-baseline) defines every feature.

## Add the separate post-August update for inference

The original F0 file stops at August 31, 2026. The supplied `updated_data_7_Nov/data/` has later stock and global observations through November 6, 2026, including all September dates in `sample_submission.csv`. To build a separate F0 file from both histories, run from the repository root:

```text
uv run python -m shared.updated_features
```

This writes `data/processed/f0_with_updates.csv`. Load it with the same `load_f0_dataset` function, or pass it to `members.zhenci.predict_e1` (its current default). The builder uses the original metadata and raw files, appends later observations, and applies the **same** shared 21-feature formulas and date rules. It does not alter `f0.csv`, the source CSVs, a fitted preprocessor, or a saved model. Updated `y` values are discarded before feature construction; on October 23, 2026, the update has `y = -1` for all 100 stocks, so those labels must not be used for scoring or training without separate resolution.

If the source update or official template is elsewhere, use `--update-dir`, `--template`, and `--output`. The builder rejects overlap with the original history and missing template keys within observed stock history. A single future template date can be represented by a key-only row; multiple future dates for the same stock require actual intervening market data. Source dates still do not prove publication before 09:00, and supplied technical-indicator settings remain undocumented; see the [central data-quality section](../../README.md#data-quality-and-timing-assumptions).
