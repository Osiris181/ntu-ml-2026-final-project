# E3: XGBoost follow-up studies

This folder contains later XGBoost validation studies and the provisional artifact tools for the training-regime study. The [member overview](../README.md) gives the experiment details and result paths. These studies retain the shared F0 source, chronological evaluation keys, five seeds where models are trained, and official scorer.

| File | Use |
|---|---|
| [`train_e3.py`](train_e3.py) | Optuna search on F0, including two E1 control trials. |
| [`train_e3_regime.py`](train_e3_regime.py) | Compare training on 100 or 50 stocks and recency weighting. |
| [`save_e3_regime.py`](save_e3_regime.py) | Refit and freeze the selected regime alternative only if it beats the E1 control. |
| [`predict_e3_regime.py`](predict_e3_regime.py) | Export a saved regime artifact through the official Phase 1 template, if an eligible artifact exists. |
| [`features_e3_relative.py`](features_e3_relative.py) | Build three market-relative features from already lagged F0 and fit their preprocessing per fold. |
| [`train_e3_feature_sets.py`](train_e3_feature_sets.py) | Compare four F0/relative/global-ablation feature sets. |
| [`evaluate_e3_threshold.py`](evaluate_e3_threshold.py) | Evaluate decision cutoffs from saved validation probabilities; no model training. |
| [`train_e3_dart.py`](train_e3_dart.py) | Compare the E1 tree booster with two DART dropout settings. |
| [`evaluate_e3_market_regime.py`](evaluate_e3_market_regime.py) | Compare fixed and prior-TAIEX-regime thresholds on saved E1 probabilities; no model training. |
| [`predict_e3_market_regime.py`](predict_e3_market_regime.py) | Apply the selected TAIEX rule to the frozen E1 model's Phase 1 probabilities and export the official template. |

Run from the repository root using `uv run python -m members.zhenci.e3.<script_name>`, for example:

```text
uv run python -m members.zhenci.e3.train_e3_feature_sets
```

The existing `results/` paths have not moved. Feature-set, threshold, and DART studies are validation only; a winning setting needs a matching frozen artifact before submission. The regime artifact scripts provide that path when a non-baseline configuration wins.

The market-regime study uses the shared F0 `taiex_return` field, which comes from the latest eligible observation strictly before each prediction date. It compares the ordinary `0.5` rule, two fixed-threshold controls, and three rules that use a higher threshold after a negative prior TAIEX return. Fixed-threshold controls use their own threshold even when TAIEX is missing; regime rules use `0.5` for missing or stale TAIEX. It also reports a walk-forward check that chooses each year's rule using only earlier validation years. Run from the repository root:

```text
uv run python -m members.zhenci.e3.evaluate_e3_market_regime
```

The default output is `results/zhenci_xgboost_e3_market_regime_v2/`. The earlier `v1` results remain on disk, but its fixed-threshold controls incorrectly used `0.5` on missing-TAIEX dates. The same-period selected score is tuned on the three CV years and is not an independent test. This script does not change the shared F0 file, retrain XGBoost, use 2026 holdout labels, or create a submission artifact.

To make a **provisional Phase 1 CSV** from the `v2` CV-selected rule, first ensure `data/processed/f0_with_updates.csv` covers every date in the official template and the saved full-sample E1 artifact exists. Then run:

```text
uv run python -m members.zhenci.e3.predict_e3_market_regime
```

The exporter checks the E1 artifact hashes, selected rule, complete template stock/date keys, and binary output. It writes `submissions/zhenci_e3_taiex_phase1.csv` without overwriting an existing file. The model and preprocessing remain frozen; only the postprocessing threshold varies by the already lagged F0 `taiex_return`. It does not upload the file. A CV improvement does not guarantee a better competition score, and the final Phase 2 artifact decision remains separate.
