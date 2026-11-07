# AGENTS.md — Coding Agent Project Context

**NTU Machine Learning Final Project, Fall 2026**  
**Protocol:** v1.1 / 2026-11-07  
**Audience:** Codex and other coding agents operating inside this repository.

> **Instruction priority:** Obey user requests and the official `final_project.pdf`. Treat this file as the team's agreed design, not as a claim that code already exists. **LOCKED** choices must be preserved. **MEMBER-OWNED** choices are intentionally flexible. **SHARED / OPEN** details must be surfaced rather than guessed. See `README.md` for the human-facing overview.

## 1. Your immediate role

The next expected agent task is to **scaffold a reproducible experiment directory and common interfaces** so four teammates can independently run their experiments.

Unless explicitly instructed, **do not** automatically train models, choose owners, connect to the competition website, make submissions, fetch live market data, generate fabricated metrics, or implement irreversible assumptions about missing timestamps. You may create clear, runnable shared code and tests for *verified* behavior when asked; create TODOs/explicit exceptions for unresolved behavior. Avoid unnecessary abstraction and dependencies.

### Team folder owners (model assignments not decided)

- `members/鄭振慈/`
- `members/葉家堯/`
- `members/龔建瑄/`
- `members/倪永珅/`

One distinct primary ML approach per member. Candidate families: Logistic Regression, Random Forest, LightGBM/XGBoost boosted-tree track, MLP. Someone is interested in boosted trees but no exact owner-model mapping has been agreed. **Do not guess who.**

## 2. Research goal and official metric — LOCKED

Each sample is one `(stock_id, prediction_date)` pair. A binary prediction `1` means adjusted close > adjusted open that trading day, otherwise `0`. Produce predictions for the `Top50@Y` IDs from the official metadata.

```python
realized_intraday_return = (adjusted_close - adjusted_open) / adjusted_open
contribution = (2 * predicted_y - 1) * realized_intraday_return
official_score = mean(contribution)
```

The **official score** is the primary optimization criterion, not accuracy, F1 or AUC. Phase 1 requires 4,200 predictions (50 stocks × 84 dates); Phase 2 requires 50 per session. The team voluntarily uses the stricter **pre-09:00 Asia/Taipei feature information cutoff** for training and both phases; Phase 1's *upload deadline* remains 2026-12-24 23:59. Phase 2 daily upload cutoff is 09:00 local time.

## 3. Data sources — verified raw ZIP structure

`fin_pred_data.zip` contains 102 entries:

- `data/market/<stock_id>.csv`: **100** stock-level files with **161,746** total stock-date observations from **2020-01-02 through 2026-08-31**.
- `data/global/global_data.csv`: date + **6** global market series, **1,506** rows.
- `data/info.json`: keys `Top50@Y` (50 competition stock identifiers), `Top100@Y` (all 100 stocks).

**Exact stock-file columns (CSV order):**

```text
date
etl:adj_close
etl:adj_open
etl:adj_high
etl:adj_low
price:成交筆數
price:成交金額
price:成交股數
foreign_investors_shareholding:全體外資及陸資持股比率
institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)
institutional_investors_trading_summary:投信買賣超股數
institutional_investors_trading_summary:自營商買賣超股數(自行買賣)
institutional_investors_trading_summary:自營商買賣超股數(避險)
RSI
SMA
MACD
ATR
BBANDS_upper
BBANDS_mid
BBANDS_lower
y
```

**Exact global header:** `date,^VIX,^IXIC,^GSPC,^TWII,^KS11,TX_OI`.

Observations from earlier data inspection: eight missing values in the stock predictors; some market dates have no same-date global record (112 distinct stock dates unmatched by calendar date); no null global cells. `y` matches `etl:adj_close > etl:adj_open` in supplied rows. Treat stock IDs as **strings**, not numerical magnitudes. Parse dates, verify uniqueness, and preserve raw source files unchanged.

**Important unresolved provenance:** global dates do not establish when each market's value became publicly available. Publication/processing delays for institutional data are unknown. Indicator period/parameter settings in the supplied RSI/SMA/MACD/ATR/Bollinger fields and the precise semantics of `TX_OI` are not documented. **Do not invent timestamps, indicator settings, or TX_OI definitions.**

## 4. Shared experiment contract — LOCKED

### Samples, stock universes, and date folds

- Main **E1 baseline** trains **one pooled model on all 100 stocks**. Evaluate each validation/test period on **only the same 50 target stocks**, identified from `info.json` (`Top50@Y`). Do not drop rows from the official target-key evaluation set.
- Fold 1: train **2020-01-01 to 2022-12-31**, validate **2023-01-01 to 2023-12-31**.
- Fold 2: train **2020-01-01 to 2023-12-31**, validate **2024-01-01 to 2024-12-31**.
- Fold 3: train **2020-01-01 to 2024-12-31**, validate **2025-01-01 to 2025-12-31**.
- Final independent holdout: train/refit **2020-01-01 through 2025-12-31** (100 stocks), test **2026-01-01 through 2026-05-31** (50 target stocks).
- Intersect date windows with **actual available trading sessions**; sort by `(date, stock_id)` for deterministic output; enforce unique `(stock_id,date)` keys.
- Split by **calendar trading date**; never randomly mix future/past stock-date rows. Any common data object for a fold must carry the same evaluation keys across all members.
- Single random seed **42**. Extra seeds are optional, not part of required base pipeline.

### Selection and test isolation

For each model/configuration, calculate official score on target-stock predictions for each validation year **separately**, then maximize the **equally weighted mean of three fold scores**:

```text
CV_score = (S_2023 + S_2024 + S_2025) / 3
```

Do **not** replace this with pooled sample-weighted score, accuracy, or validation on training stocks. Save all individual fold scores. Hyperparameters, feature variants, optional threshold selection and other tuning use CV folds, **not** final Jan–May 2026 test. Freeze choices before single final holdout evaluation. Do not reselect after observing holdout outcomes.

### Prediction-time cutoff and leakage safety

For target date `t`, all features must have been **published strictly before 09:00 Asia/Taipei on `t`**, including simulated historical predictions and Phase 1. In particular:

- Target-day adjusted OHLC, intraday return, realized score, `y`, end-of-day volume and later technical indicators are **not** predictors.
- Lag/rolling calculations operate *within stock*, using only previous completed/published trading sessions. The index `t-1` is a **trading-session** index.
- Global features require a verified **as-of observation availability timestamp**; do not blindly merge by same-date strings, assume overnight market publication without evidence, or forward-fill across arbitrary gaps.
- Fitted preprocessing objects and all learned feature transformations must be fit on that fold's **training rows only**. Refit appropriately on 2020–2025 for the final test.
- This time-alignment issue is **SHARED / OPEN**. If a timestamp cannot be proven, implement an explicit verification TODO/error or a documented verified-safe lag only after approval; do not claim an unverified reconstruction is leakage-free.

### E1 and E2

- **E1 (controlled):** identical F0 source information, shared baseline preprocessing, same folds, seed and evaluation keys for all four distinct algorithms.
- **E2 (member-optimized):** individual members may try extra lagged/causal features, alternative preprocessing, feature selection and tuning strategies, but must preserve the same cutoff, fold/test keys, scorer and result schema. Compare each E2 against its corresponding E1.
- **MEMBER-OWNED:** hyperparameter search algorithm (Grid Search/Random Search/Optuna, etc.), hyperparameter ranges, trial budget, E2 feature sets, optional ablation studies and model-specific interpretability. **Log choices and compute cost.** Different optimization budgets may explain some observed differences.
- **Optional individual study:** train 100 stocks vs. train only 50 target stocks, **evaluating on the same 50 target stocks** in both conditions. Initial controlled ablation holds model/features/config/seed fixed while varying the training universe. This ablation is *not* a team-wide replacement for 100-stock E1.

## 5. Exact F0 baseline: 21 logical feature fields — LOCKED

Conventions: `open`, `close`, `high`, `low` are the **adjusted** stock prices. `t-1`, `t-2`, etc. refer to preceding **per-stock trading sessions** already available by the cutoff. For a 20-session window, use exactly `t-20 ... t-1` inclusive. Compute from original stock time series before slicing into folds **only when mathematically causal**; fitted transformations still use fold training data alone. Replace undefined ratios in baseline preprocessing rather than dropping rows. For globals, "latest" means *latest verified-available observation before cutoff*.

| # | Baseline name | Source / calculation |
|---:|---|---|
| 1 | `intraday_return_1d` | `close[t-1] / open[t-1] - 1` |
| 2 | `daily_return_1d` | `close[t-1] / close[t-2] - 1` |
| 3 | `momentum_5d` | `close[t-1] / close[t-6] - 1` |
| 4 | `price_range_1d` | `(high[t-1] - low[t-1]) / close[t-1]` |
| 5 | `relative_volume_20d` | `volume[t-1] / mean(volume[t-20:t-1])` |
| 6 | `relative_amount_20d` | `amount[t-1] / mean(amount[t-20:t-1])` |
| 7 | `trade_size_ratio_20d` | `(volume/trade_count)[t-1] / mean((volume/trade_count)[t-20:t-1])` |
| 8 | `foreign_ownership` | `foreign_ownership_percent[t-1] / 100` |
| 9 | `foreign_net_buy_ratio` | `foreign_net_buy_shares[t-1] / volume[t-1]` |
| 10 | `trust_net_buy_ratio` | `trust_net_buy_shares[t-1] / volume[t-1]` |
| 11 | `rsi_normalized` | `RSI[t-1] / 100` |
| 12 | `sma_distance` | `close[t-1] / SMA[t-1] - 1` |
| 13 | `macd_normalized` | `MACD[t-1] / close[t-1]` |
| 14 | `atr_normalized` | `ATR[t-1] / close[t-1]` |
| 15 | `bb_width` | `(BBANDS_upper[t-1] - BBANDS_lower[t-1]) / BBANDS_mid[t-1]` |
| 16 | `vix_normalized` | `latest_verified_available(^VIX) / 100` |
| 17 | `nasdaq_return` | `latest_verified_available(^IXIC) / preceding_available(^IXIC) - 1` |
| 18 | `sp500_return` | `latest_verified_available(^GSPC) / preceding_available(^GSPC) - 1` |
| 19 | `taiex_return` | `latest_verified_available(^TWII) / preceding_available(^TWII) - 1` |
| 20 | `stock_id` | Categorical string stock ID from file/metadata, **not an ordinal numeric variable** |
| 21 | `day_of_week` | Prediction trading date weekday, categorical |

Raw source-column mapping for formulas: `close`=`etl:adj_close`; `open`=`etl:adj_open`; `high`=`etl:adj_high`; `low`=`etl:adj_low`; `trade_count`=`price:成交筆數`; `amount`=`price:成交金額`; `volume`=`price:成交股數`; `foreign_ownership_percent`=`foreign_investors_shareholding:全體外資及陸資持股比率`; `foreign_net_buy_shares`=`institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)`; `trust_net_buy_shares`=`institutional_investors_trading_summary:投信買賣超股數`.

**F0 = 21 logical fields**, not necessarily 21 numeric encoded columns; `stock_id` and `day_of_week` expand under one-hot encoding. F0 includes normalized/relative financial values and two categorical context fields. No additional raw unnormalized price/volume predictors in E1. Raw originals remain available to members' E2 experiments.

## 6. Baseline preprocessing: SAME for all algorithms — LOCKED

The following applies to **E1** across all four methods:

1. Detect division by zero and undefined numerical calculations; convert `+inf` and `-inf` to `NaN`. Do not silently drop stock-date observations.
2. Numeric F0 columns: **median imputation fitted using training rows only**, followed by **StandardScaler fitted on training rows only**. Apply the fitted objects to validation/test.
3. Categorical F0 columns: replace feature missingness with a literal `Unknown`, then apply a **shared one-hot encoding policy** learned from training rows. Handle unseen categories deterministically (`handle_unknown='ignore'` is a viable implementation).
4. **No baseline winsorization, clipping, outlier removal or feature selection.** Model-specific preprocessing choices belong in **E2**, not E1.
5. Keep `stock_id` and `date` as immutable row keys, even if `stock_id` also enters F0 as a categorical predictor. A missing key is a **data integrity error**; do not fill a missing key with `Unknown`. Likewise, missing historical training labels or evaluation realized returns require explicit handling/diagnostics, not silent relabeling.
6. Strictly fit one preprocessing instance **per training fold**. For final holdout refit on all 2020–2025 training rows. Preserve the exact 50-stock validation/test keys for every model.

**Shared edge-case policy still OPEN:** all-NaN numeric feature within a training fold, missing global series with no proven safe timestamp, technical indicators unavailable for replay, and maximum staleness. Do not silently substitute an arbitrary median. Prefer a clear exception/TODO and let the shared-infrastructure owner decide.

**Classifier threshold:** initial convention is `p(up) > 0.5 -> 1`, otherwise `0`. Decision threshold optimization is allowed in E2/CV only and must not use holdout labels. Do not assume all algorithms must expose calibrated probabilities.

## 7. Recommended scaffold and file contracts

The user intends to use Codex to **create the experiment directory**. A sensible skeleton, when requested:

```text
README.md
AGENTS.md
configs/
data/raw/
data/processed/
data/splits/
shared/data_loader.py
shared/features.py
shared/preprocessing.py
shared/splits.py
shared/evaluation.py
shared/submission.py
members/鄭振慈/
members/葉家堯/
members/龔建瑄/
members/倪永珅/
results/
submissions/
tests/
requirements.txt
.gitignore
```

You may add `__init__.py`, minimal CLI/examples/configs, and lightweight tests if asked. **Do not assign specific model scripts to specific members before assignments.** Prefer relative project paths and `pathlib`; treat raw data as separately placed input; add helpful errors when expected input files are absent.

**Common output fields (proposed interface):**

- Result row: `member`, `model`, `experiment` (`E1` or `E2`), `config_id`, `fold`, `seed`, `official_score`, `accuracy`, `n_eval`, `train_seconds`, `predict_seconds`.
- Validation prediction row: `member`, `model`, `experiment`, `config_id`, `fold`, `stock_id`, `date`, `predicted_y`, optional `p_up`.
- Configuration file: feature set/version, optimizer method, number of trials, hyperparameters, preprocessing choices, training universe, seed, dependency versions, split version.

These are internal team conventions, **not** official competition submission schemas. The shared `evaluation.py` must validate binary outputs, unique and complete evaluation keys, alignment with target stock-date rows, and the adjusted-open denominator; never give different members a different set of evaluated dates.

**Minimum unit/integration tests to prioritize when coding is requested:**

1. Dates and target IDs are parsed exactly and duplicate stock-date keys raise an error.
2. Three chronological folds have disjoint training/validation dates with no temporal inversion.
3. Each validation/test fold has the same target-stock/date key manifest for all approaches.
4. No same-date target-day OHLC, `y` or return columns enter baseline features; rolling/lag tests on tiny synthetic timelines verify no future peeking.
5. Training-fold-only median/scaling/encoding: validation perturbations cannot change fitted parameters.
6. Official score matches a small hand-worked example with both up and down predictions; score is averaged per required prediction.
7. Time-zone/as-of tests should be added once validated timestamp metadata/policy exists; do not mark them passing with made-up availability times.

## 8. Submission and phase-specific integrity

Generate both phases from their **actual phase/session-specific** `sample_submission.csv` templates, preserving header spelling, stock ID order and exact dates. Validate every prediction is `0` or `1`; reject missing/duplicated/reordered/additional stock IDs. Phase 1: **4,200 predictions**. Phase 2: **50 predictions**. Daily Phase 2 only counts the latest valid file submitted before 09:00 Taiwan time. Confirm upload in Submission Status.

Archive supplied data only through **2026-08-31**: obtaining reproducible **September–December live inputs** and deciding update/retraining schedule are **SHARED / OPEN operational tasks**. Do not claim trained or deployed Phase 2 readiness while these are unresolved.

## 9. Ownership matrix — do not confuse these

| Shared/fixed | Independent/member-owned | Shared/open to verify |
|---|---|---|
| Target and official scorer | Model selection after team assignment | Global-market as-of timestamps |
| 100-stock pooled E1 train and 50-stock eval | Hyperparameter optimizer and tuning budget | Publication time of institutional features |
| Same chronological folds and test | E2 feature engineering and preprocessing | Technical-indicator reproducibility |
| Fixed seed `42` | E2 feature selection/ablation | Missing/stale source edge cases |
| Baseline 21 F0 fields and shared E1 preprocessing | Optional 100-vs-50 training ablation | Results schema implementation and secondary metrics |
| Historical pre-09:00 cutoff and test isolation | Model-specific diagnostics | Live input updates, retraining, submission integration |

If a user later changes a **LOCKED** item, update `README.md` and `AGENTS.md` together and record the protocol revision. If only a member-owned experiment changes, do not modify the central baseline contract.

**Avoid making unsupported claims about model superiority, leakage-free data, up-to-date global timestamps, or competition performance. Report what has actually been verified.**
