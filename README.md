# Intraday Direction Prediction for Taiwanese Stocks

This repository supports the Fall 2026 NTU Machine Learning final project. The task is to predict whether each of 50 Taiwanese stocks will finish a trading day above its adjusted opening price. The repository contains the shared experiment infrastructure and four member workspaces. An XGBoost E1 runner is prepared for 鄭振慈; it has not been run, and the repository does not yet contain trained models or measured model results.

The [team workflow guide](context/README.md) explains how members use the current pipeline. The [v1.1 protocol record](context/protocol-v1.1.md) preserves the original decisions and course requirements. The official [project specification](final_project.pdf) takes precedence.

## Prediction task and score

One sample is a `(stock_id, date)` pair. The label is `1` when adjusted close exceeds adjusted open; otherwise it is `0`, including no change.

```text
r = (adjusted_close - adjusted_open) / adjusted_open
official_score = mean((2 × predicted_y - 1) × r)
```

The mean covers the required target stock/date predictions. Higher scores are better. Accuracy is a secondary diagnostic; configuration selection uses the official score.

## Data and information cutoff

The supplied data contains 100 stock CSV files with 161,746 stock/date observations from 2020-01-02 through 2026-08-31, a global-market CSV, and `data/info.json`. The metadata identifies all 100 supplied stocks (`Top100@Y`) and the 50 targets (`Top50@Y`). The shared loader validates the source schemas, dates, binary labels, and unique stock/date keys.

The team's prediction cutoff is **strictly before 09:00 Asia/Taipei on the prediction date**. The shared F0 builder enforces conservative, fixed historical date lags. Target-day prices, volume, indicators, labels, and realized returns are excluded from its output.

### Data quality and timing assumptions

The raw CSVs have observation dates, **not externally verified publication timestamps**. The team has therefore adopted one central rule for all four approaches: use the previous completed stock trading session (`t-1`) for price, trading activity, and supplied technical indicators; use `t-2` for institutional values and the volume denominators in their ratios; use each global series' latest nonmissing observation dated strictly before the prediction date, only if it is at most **five calendar days** old. Missing or stale feature values remain `NaN` for the existing fold-fitted preprocessing. This date policy is a conservative assumption, **not proof that every value was public before 09:00**.

The stock CSVs already contain RSI, SMA, MACD, ATR, and Bollinger Band values, which F0 uses directly. Their calculation settings are undocumented, so reproducing them consistently for new data remains unresolved. Actual publication delays for stock, institutional, technical, and global fields also remain unverified. The competition is not assumed to supply updated market data automatically.

## Shared F0 baseline

The controlled E1 comparison uses the same **21 logical features** for every approach. Stock prices in these formulas are adjusted. `t-1` and `t-2` refer to prior completed sessions **for that stock**; a 20-session window contains `t-20` through `t-1`, inclusive. “Latest” global values follow the date and five-day rule in the data-quality section.

| Group | Feature | Definition |
|---|---|---|
| Price | `intraday_return_1d` | `close[t-1] / open[t-1] - 1` |
| Price | `daily_return_1d` | `close[t-1] / close[t-2] - 1` |
| Price | `momentum_5d` | `close[t-1] / close[t-6] - 1` |
| Price | `price_range_1d` | `(high[t-1] - low[t-1]) / close[t-1]` |
| Trading | `relative_volume_20d` | `volume[t-1] / mean(volume[t-20:t-1])` |
| Trading | `relative_amount_20d` | `amount[t-1] / mean(amount[t-20:t-1])` |
| Trading | `trade_size_ratio_20d` | `(volume / trade_count)[t-1] / mean((volume / trade_count)[t-20:t-1])` |
| Institutional | `foreign_ownership` | `foreign_ownership_percent[t-2] / 100` |
| Institutional | `foreign_net_buy_ratio` | `foreign_net_buy_shares[t-2] / volume[t-2]` |
| Institutional | `trust_net_buy_ratio` | `trust_net_buy_shares[t-2] / volume[t-2]` |
| Technical | `rsi_normalized` | `RSI[t-1] / 100` |
| Technical | `sma_distance` | `close[t-1] / SMA[t-1] - 1` |
| Technical | `macd_normalized` | `MACD[t-1] / close[t-1]` |
| Technical | `atr_normalized` | `ATR[t-1] / close[t-1]` |
| Technical | `bb_width` | `(BBANDS_upper[t-1] - BBANDS_lower[t-1]) / BBANDS_mid[t-1]` |
| Global | `vix_normalized` | `latest_eligible(^VIX) / 100` |
| Global | `nasdaq_return` | `latest_eligible(^IXIC) / preceding_observed(^IXIC) - 1` |
| Global | `sp500_return` | `latest_eligible(^GSPC) / preceding_observed(^GSPC) - 1` |
| Global | `taiex_return` | `latest_eligible(^TWII) / preceding_observed(^TWII) - 1` |
| Context | `stock_id` | Categorical stock identifier |
| Context | `day_of_week` | Categorical prediction-date weekday |

These are 21 logical inputs, not 21 final numeric columns: one-hot encoding expands the two categorical fields. E1 adds no raw price or volume levels as predictors.

### Reusable shared F0 dataset

`shared/features.py` applies these rules once for all 100 stocks. The generated `data/processed/f0.csv` has **161,746 rows** and 22 columns: `stock_id`, `date`, and the other 20 F0 fields. It preserves rows with missing history or stale global values; it contains no `y` or target-day OHLC columns. Members load it directly with `load_f0_dataset`, then use the existing chronological fold keys to select rows. Training labels and evaluation prices come separately from the unchanged raw market data.

```text
uv run python -m shared.features --data-dir data --output data/processed/f0.csv
```

```python
from shared.features import load_f0_dataset

f0 = load_f0_dataset("data/processed/f0.csv")
```

The generated CSV is rebuildable and gitignored. Fit the existing `BaselinePreprocessor` only on each fold's training rows; do not impute or scale the shared dataset before splitting.

## Model comparison and evaluation

The project compares **four meaningfully different ML approaches** on common target keys. Each approach has two experiment settings:

- **E1, controlled baseline:** shared F0 inputs, preprocessing, folds, and scorer.
- **E2, optimized approach:** the approach owner may change features, preprocessing, and tuning while preserving the same information cutoff, evaluation keys, folds, and scorer.

The shared E1 preprocessing converts infinite numeric values to missing, imputes numeric fields with training-fold medians, and standardizes using training-fold means and population standard deviations. Missing categorical feature values become `Unknown`; one-hot categories are learned from the training fold. Unseen evaluation categories are handled deterministically. An all-missing numeric training feature raises an error. E1 has no default clipping, winsorization, or feature selection.

Training uses all 100 supplied stocks. Validation and final testing use only the 50 targets. Every approach must predict the same target stock/date rows. Preprocessing is fitted separately for each training window.

| Run | Training dates | Evaluation dates |
|---|---|---|
| Fold 1 | 2020–2022 | 2023 |
| Fold 2 | 2020–2023 | 2024 |
| Fold 3 | 2020–2024 | 2025 |
| Independent holdout | 2020–2025 | 2026-01-01 through 2026-05-31 |

Windows are intersected with trading rows present in the data. The shared seed is `42`. Configuration selection uses the **equal-weight mean** of the three yearly validation official scores: `(score_2023 + score_2024 + score_2025) / 3`. The Jan–May 2026 holdout is reserved for evaluation after choices are frozen.

## Phase 2 frozen-model strategy

This is the team's **agreed deployment strategy**, not an implemented training or daily inference workflow.

1. **Train and select:** run the existing three expanding validation folds. Compare configurations by the equal-weight mean official score, with model-specific search methods and budgets chosen by each member. Keep the Jan–May 2026 holdout isolated and evaluate it only after the configuration is fixed.
2. **Refit and freeze:** retrain the chosen configuration on the agreed eligible historical training data. Save the trained model, its fitted preprocessing pipeline, the exact F0 feature definitions and order, and its binary prediction threshold or decision rule. The eligible deployment-refit window and artifact format still need a team decision; do not infer them from the current CSV end date.
3. **Run daily inference:** obtain newly available historical market inputs, build the same 21 F0 fields under the shared lag and cutoff rules, apply the **frozen** preprocessing pipeline, and run the **frozen** model and threshold to produce 50 binary predictions. Export them with that day's actual official Phase 2 submission template.

Daily updates change **input feature values only**. They do not refit preprocessing, change model parameters or threshold, or trigger tuning. Live input acquisition and validation remain implementation tasks; see the data-quality section for unresolved timing and indicator assumptions.

Phase 2 inference is defined around F0. If the best validation configuration is an E2 pipeline that requires additional features, the team must resolve its deployment compatibility before freezing an artifact; feature inputs cannot be silently removed.

## Submission file format

The supplied `sample_submission.csv` is the Phase 1 template: 50 `id` rows and 84 date columns, for 4,200 binary predictions. `shared/submission.py` accepts long-form model output with `stock_id`, `date`, and `predicted_y`, then writes the template's exact stock order and date headers. It rejects missing, extra, duplicate, or nonbinary predictions. The same utility reads a future phase/session template directly; it does not invent that template's dates or order. No prediction file has been generated from a trained model yet.

## Repository map

| Path | Contents |
|---|---|
| `data/market/`, `data/global/`, `data/info.json` | Supplied historical inputs and stock lists |
| `shared/data_loader.py`, `shared/features.py` | Input validation and centralized, conservative F0 construction |
| `shared/splits.py`, `configs/protocol.json` | Chronological folds and seed |
| `shared/preprocessing.py` | Shared E1 preprocessing |
| `shared/evaluation.py` | Prediction-key checks, official score, accuracy, and CV score |
| `shared/submission.py`, `sample_submission.csv` | Validated template-based submission export and the supplied Phase 1 template |
| `members/` | Four personal experiment directories; 鄭振慈 has a prepared XGBoost E1 runner |
| `tests/` | Shared integrity, feature, split, preprocessing, and scoring tests |

## Local checks

Use [`uv`](https://docs.astral.sh/uv/) from the repository root. Dependencies are declared in `pyproject.toml`; `uv run` creates or uses the project environment and lockfile:

```text
uv run python -m unittest discover -s tests -v
```

The prepared [XGBoost E1 runner](members/鄭振慈/README.md) has its own usage instructions. It has not been executed or used to select a model.

The shared F0 CSV has been generated from the supplied historical data. Its lag rules are consistent across members; the data-quality assumptions above remain unverified externally.
