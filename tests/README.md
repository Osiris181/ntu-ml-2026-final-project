# Shared pipeline tests

[`test_shared.py`](test_shared.py) contains automated checks for source integrity, stock-session lags, independent global-series alignment and staleness, complete F0 rows, consistent feature columns, chronological splits, fold-fitted preprocessing, official scoring, and submission formatting.

[`test_updated_features.py`](test_updated_features.py) checks that later source labels are discarded, overlapping dates are rejected, a future template date uses only earlier values, and missing historical template keys fail explicitly.

[`test_zhenci_e2.py`](test_zhenci_e2.py) checks the XGBoost E2 configuration selection rule and its deterministic tie handling without running training.

[`test_zhenci_e2_features.py`](test_zhenci_e2_features.py) checks that member-owned F1 fields use only earlier stock sessions and that their added preprocessing is fitted on training rows while F0 preprocessing remains identical.

[`test_zhenci_e2_weighted.py`](test_zhenci_e2_weighted.py) checks training-return weight scaling, flat-session handling, and historical label consistency without fitting a model.

[`test_zhenci_e2_return_regression.py`](test_zhenci_e2_return_regression.py) checks the XGBoost E2 return target and conversion of regression output into binary predictions without fitting a model.

[`test_zhenci_e3_regime.py`](test_zhenci_e3_regime.py) checks the E3 training-universe filter, fold-cutoff recency weights, and refusal to save an alternative that did not beat the E1 control. It does not fit a model.

[`test_zhenci_e3_variants.py`](test_zhenci_e3_variants.py) checks market-relative features, fold-fitted preprocessing and restoration, and chronological threshold selection without fitting a model.

[`test_zhenci_e3_market_regime.py`](test_zhenci_e3_market_regime.py) checks the prior-TAIEX threshold rule, its missing-value fallback, selection using only earlier validation years, and exporter rejection of mismatched saved rules. It does not fit a model.

[`test_zhenci_e4_features.py`](test_zhenci_e4_features.py) checks E4's completed-session rolling windows, missing prices, historical/updated feature consistency, exact keys, and training-fold-only F2 preprocessing.

[`test_zhenci_e4_runner.py`](test_zhenci_e4_runner.py) checks the pinned E1 configuration and E3 rule, saved control probabilities and exact keys, and score reproduction guards without fitting a model.

Run from the repository root:

```text
uv run python -m unittest discover -s tests -v
```

These tests verify the implemented date rules. They cannot verify the raw data's actual publication times or the undocumented settings used to calculate supplied technical indicators; those assumptions are recorded in the [project README](../README.md#data-quality-and-timing-assumptions).
