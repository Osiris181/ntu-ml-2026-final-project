# Shared pipeline tests

[`test_shared.py`](test_shared.py) contains automated checks for source integrity, stock-session lags, independent global-series alignment and staleness, complete F0 rows, consistent feature columns, chronological splits, fold-fitted preprocessing, official scoring, and submission formatting.

Run from the repository root:

```text
uv run python -m unittest discover -s tests -v
```

These tests verify the implemented date rules. They cannot verify the raw data's actual publication times or the undocumented settings used to calculate supplied technical indicators; those assumptions are recorded in the [project README](../README.md#data-quality-and-timing-assumptions).
