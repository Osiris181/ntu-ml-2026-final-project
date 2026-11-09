# E7: Historical payoff cutoffs for the frozen E1 probability model

E5's feature removals and E6's inner early stopping did not beat the E3 TAIEX-rule control. E6 often selected one tree and predicted almost every stock down. E7 tests a decision rule aimed at the **official return score** while keeping the existing full-sample E1 XGBoost probabilities and shared F0 features.

## Idea in plain language

The ordinary `0.5` cutoff asks whether an up day is more likely than a non-up day. The official score also depends on the **size** of gains and losses. On each fold's training rows only, E7 estimates:

- the average gain on days with adjusted close above adjusted open;
- the average loss magnitude on all other days, including flat days with zero loss.

If the model's up probability is `p`, the estimated return is `p × average_gain − (1 − p) × average_loss`. Predict up when that estimate is positive. This gives the cutoff:

```text
average_loss / (average_gain + average_loss)
```

The formula assumes the probability and historical payoff estimates transfer to the next year; the experiment checks whether they do. The model still uses the same pre-09:00 F0 inputs. Historical returns are used only to **fit the cutoff on earlier training dates**, never as target-date features.

## Run

From `final_project`:

```text
uv run python -m members.zhenci.e7.evaluate_e7
```

The default output is `results/zhenci_xgboost_e7_payoff_v1/`. It refuses to overwrite an existing directory. Optional flags allow alternate source and output paths. The script reads original `data/processed/f0.csv`, original market history, saved E1 validation probabilities, and the selected E3 TAIEX rule. It does **not** train another XGBoost model.

## Compared decision rules

| Arm | Cutoff source |
|---|---|
| E3 control | Existing fixed prior-TAIEX rule: `0.525` / `0.475` / `0.5` |
| E1 control | Fixed `0.5` |
| Pooled payoff | One payoff cutoff learned from all 100 training stocks |
| TAIEX-regime payoff | Separate training-payoff cutoff for negative, nonnegative, or missing prior TAIEX; pooled fallback if a state has no training rows |
| Stock payoff | Each stock's historical payoff cutoff, with each up and non-up average pulled toward the pooled average by **100 pseudo-observations per class** |

All five seeds and the same three expanding validation folds are evaluated on the same 50 target stocks with the official score. Payoff cutoffs for 2023 use only 2020–2022 returns; 2024 uses only 2020–2023; 2025 uses only 2020–2024. No cutoff uses its own validation year's returns. The result directory contains `summary.json`, `fold_metrics.csv`, `payoff_cutoffs.csv`, and `selected_validation_predictions.csv`.

The incumbent to beat is E3's mean CV score of about `0.00064050`. A read-only preliminary check of the stock-payoff rule on these **same** validation years averaged about `0.000709`; it improved 2023 and 2025 but slightly reduced 2024. E7 records this as **exploratory**, not independent confirmation. Several earlier member experiments already used these CV years, so even a reproduced gain needs cautious interpretation and a separate holdout check before a new artifact or submission. The 2026 holdout and Phase 1 outcome are not used by this runner.

## Selected result and independent check

The completed E7 validation run selected `stock_payoff_shrunk_100`: mean official score `0.00070881`, compared with E3's `0.00064050`. It beat E3 in 2023 and 2025, but not in 2024. All five seeds produced identical scores because the underlying E1 probabilities are deterministic. The validation improvement is exploratory; it is not a new competition submission result.

The selected stock payoff rule and its 100 pseudo-observations per class are now fixed for a **single** Jan–May 2026 holdout check. From `final_project`, run:

```text
uv run python -m members.zhenci.e7.evaluate_holdout
```

No flags are needed. This script requires the existing `results/zhenci_e1_full_pre_holdout_artifact/`, which was trained through December 2025, and original `data/processed/f0.csv`. It fits stock payoff cutoffs only on 2020–2025 training returns, applies the frozen E1 model and preprocessor to 2026 holdout F0, then reports E7 alongside the fixed E1 and E3 controls. It does not train or tune a model. The default output is `results/zhenci_xgboost_e7_holdout_v1/` with `summary.json`, `holdout_predictions.csv`, and `payoff_cutoffs.csv`. It refuses to overwrite that directory. Do not use holdout results to change the selected cutoff policy or model settings.

The completed holdout check scored `0.00090431` for E7, below E3's `0.00144758` and E1's `0.00152019`. E7's validation gain did not generalize to this period. These results do not change E7's preselected rule.

## Export an exploratory Phase 1 submission

[`predict_e7.py`](predict_e7.py) applies that same frozen stock payoff rule to the official Phase 1 template. It uses the saved E1 model and preprocessor, fits stock payoff cutoffs from original 2020–2025 training returns, and reads the already generated `data/processed/f0_with_updates.csv` for template-date features. No new model is trained. Run from `final_project`:

```text
uv run python -m members.zhenci.e7.predict_e7
```

The default output is `submissions/zhenci_e7_stock_payoff_phase1.csv`. The exporter checks the locked E7 selection, frozen artifact hashes, complete 4,200 template keys, and binary predictions; it refuses to overwrite an existing CSV. It only creates a local file. Uploading it is a separate competition action. Competition feedback on this additional period can describe E7's performance there, but must not be used to retune the rule or replace the independent holdout result.
