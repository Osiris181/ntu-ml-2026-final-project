# AGENTS.md — Coding Agent Project Context

**NTU Machine Learning Final Project, Fall 2026**  
**Protocol:** v1.2 E1 seed revision plus centralized F0 lag policy and Phase 2 frozen-model strategy
**Audience:** Codex and other coding agents operating inside this repository.

> **Instruction priority:** Obey user requests and the official `final_project.pdf`. Treat this file as the team's agreed design, not as a claim that code already exists. **LOCKED** choices must be preserved. **MEMBER-OWNED** choices are intentionally flexible. **SHARED / OPEN** details must be surfaced rather than guessed. See `context/README.md` for the current team workflow, `context/protocol-v1.2.md` for the seed revision, `context/protocol-v1.1.md` for the original protocol, and `README.md` for the public overview.

## 1. Repository state and your role

The repository has shared loaders, a generated F0 dataset, chronological splits, E1 preprocessing, scoring and template-based submission export utilities, tests, and four member workspaces. The project uses `uv` and `pyproject.toml` for dependencies. Members use `data/processed/f0.csv` or regenerate it with `uv run python -m shared.features`; do not implement separate E1 lag rules. The current team workflow is in `context/README.md`; the original protocol is archived at `context/protocol-v1.1.md`. Preserve these shared contracts as members add their own experiments.

Unless explicitly instructed, **do not** automatically train models, choose owners, connect to the competition website, make submissions, fetch live market data, generate fabricated metrics, or implement irreversible assumptions about missing timestamps. You may create clear, runnable shared code and tests for *verified* behavior when asked; create TODOs/explicit exceptions for unresolved behavior. Avoid unnecessary abstraction and dependencies.

### Team folder owners

- `members/鄭振慈/`
- `members/葉家堯/`
- `members/龔建瑄/`
- `members/倪永珅/`

One distinct primary ML approach per member. 鄭振慈 has chosen XGBoost for the boosted-tree track; the other primary model assignments are still open. Candidate families for those remaining tracks include Logistic Regression, Random Forest, and MLP. Do not assign the remaining models without a team decision.

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

### Data quality and timing assumptions — centralized, not externally verified

The raw CSVs provide observation dates but no verified publication timestamps. The team now applies the **same conservative date policy** to all four approaches: price, trading activity, and supplied technical indicators use the previous completed per-stock trading session (`t-1`); institutional values and their volume denominators use `t-2`; each global series independently uses its latest nonmissing observation **strictly dated before** the prediction date, with a **five-calendar-day** maximum age. Missing or stale values stay missing for fold-fitted E1 preprocessing. No stock/date row is dropped for ordinary feature missingness.

This policy is an agreed proxy for the pre-09:00 cutoff, **not external proof of actual publication time**. Stock, institutional, and technical processing delays, global market timing across time zones, and source revisions remain unverified. Historical F0 uses the supplied RSI/SMA/MACD/ATR/Bollinger values directly, but their period/parameter settings are undocumented and must be resolved before reproducing them on new data. The meaning of `TX_OI` is also undocumented and it is not an F0 feature. Do not invent timestamps or indicator settings, and do not claim the proxy has been independently certified as leakage-free.

## 4. Shared experiment contract — LOCKED

### Samples, stock universes, and date folds

- Main **E1 baseline** trains **one pooled model on all 100 stocks**. Evaluate each validation/test period on **only the same 50 target stocks**, identified from `info.json` (`Top50@Y`). Do not drop rows from the official target-key evaluation set.
- Fold 1: train **2020-01-01 to 2022-12-31**, validate **2023-01-01 to 2023-12-31**.
- Fold 2: train **2020-01-01 to 2023-12-31**, validate **2024-01-01 to 2024-12-31**.
- Fold 3: train **2020-01-01 to 2024-12-31**, validate **2025-01-01 to 2025-12-31**.
- Final independent holdout: train/refit **2020-01-01 through 2025-12-31** (100 stocks), test **2026-01-01 through 2026-05-31** (50 target stocks).
- Intersect date windows with **actual available trading sessions**; sort by `(date, stock_id)` for deterministic output; enforce unique `(stock_id,date)` keys.
- Split by **calendar trading date**; never randomly mix future/past stock-date rows. Any common data object for a fold must carry the same evaluation keys across all members.
- Required E1 seeds: **42, 64, 128, 256, 512**, for each model configuration on each of the same three folds. Seed 42 remains the reference seed. Do not choose the best seed after seeing validation results.

### Selection and test isolation

For each model/configuration and seed, calculate official score on target-stock predictions for each validation year **separately**. Compute each seed's **equally weighted mean of three fold scores**, then compare configurations by the mean of those five seed-level CV scores:

```text
CV_score = (S_2023 + S_2024 + S_2025) / 3
E1_selection_score = mean(CV_score_seed_42, CV_score_seed_64, CV_score_seed_128, CV_score_seed_256, CV_score_seed_512)
```

Do **not** replace this with pooled sample-weighted score, accuracy, or validation on training stocks. Save all 15 seed/fold scores, each seed's CV score, and their mean and standard deviation. Five seeds measure sensitivity to randomization; they do not guarantee robustness or remove validation-period luck. Deterministic model configurations may return identical results for all five seeds; report that rather than claiming independent variation. Hyperparameters, feature variants, optional threshold selection and other tuning use CV folds, **not** final Jan–May 2026 test. Freeze choices before single final holdout evaluation. Do not reselect after observing holdout outcomes. The final deployment-refit seed or ensemble policy remains **SHARED / OPEN**.

### Prediction-time cutoff and leakage safety

For target date `t`, all features must have been **published strictly before 09:00 Asia/Taipei on `t`**, including simulated historical predictions and Phase 1. In particular:

- Target-day adjusted OHLC, intraday return, realized score, `y`, end-of-day volume and later technical indicators are **not** predictors.
- Lag/rolling calculations operate *within stock*. Use `t-1` for price, trading activity, and supplied technical indicators; use `t-2` for institutional fields **and the volume denominator in their ratios**. These are trading-session indices.
- Global features use the latest nonmissing observation dated strictly before `t`, separately for each series; reject observations older than five calendar days. Never join same-date global rows or forward-fill across longer gaps.
- Fitted preprocessing objects and all learned feature transformations must be fit on that fold's **training rows only**. Refit appropriately on 2020–2025 for the final test.
- The central data-quality section records what this conservative date policy does and does not establish. Do not turn its unverified publication assumptions into claims of externally proven availability.

### E1 and E2

- **E1 (controlled):** identical F0 source information, shared baseline preprocessing, same folds, five-seed schedule and evaluation keys for all four distinct algorithms.
- **E2 (member-optimized):** individual members may try extra lagged/causal features, alternative preprocessing, feature selection and tuning strategies, but must preserve the same cutoff, fold/test keys, scorer and result schema. Compare each E2 against its corresponding E1.
- **MEMBER-OWNED:** hyperparameter search algorithm (Grid Search/Random Search/Optuna, etc.), hyperparameter ranges, trial budget, E2 feature sets, optional ablation studies and model-specific interpretability. **Log choices and compute cost.** Different optimization budgets may explain some observed differences.
- **Optional individual study:** train 100 stocks vs. train only 50 target stocks, **evaluating on the same 50 target stocks** in both conditions. Initial controlled ablation holds model/features/config/seed fixed while varying the training universe. This ablation is *not* a team-wide replacement for 100-stock E1.

### Phase 2 frozen-model inference — agreed strategy

Keep **training**, **selection**, **deployment refitting**, and **daily inference** distinct:

1. Train and select a configuration with the existing three expanding validation folds and required five E1 seeds. Compare by the equal-weight mean of the five seed-level CV scores. Model-specific optimizer, search range, and trial budget remain **MEMBER-OWNED**. Preserve the independent Jan–May 2026 holdout: evaluate it once after configuration choices are frozen, never use it to select or retune.
2. After selection, retrain the chosen configuration on the agreed eligible historical training data for deployment. Save and freeze the trained model parameters, fitted preprocessing objects, exact feature definitions and column order, and binary prediction threshold or decision rule. The deployment-refit date window, final seed or ensemble policy, and artifact storage/serialization format are **SHARED / OPEN**; do not assume that all rows through the latest CSV date are eligible.
3. For each Phase 2 session, obtain and validate newly available historical input data. Build the same **21 F0 fields** for the 50 target stocks under the central `t-1`/`t-2`/strict-prior-global rules and the **09:00 Asia/Taipei** prediction cutoff. Apply the frozen preprocessing objects without fitting them again. Use the frozen model and threshold to produce exactly 50 binary predictions, then export them from that session's actual official submission template.
4. **Do not retrain, tune, select a new threshold, update preprocessing statistics or category vocabulary, or modify model parameters during daily inference.** Updating the input feature values does not modify the saved model.

The competition is **not** assumed to supply updated market data automatically. Reproducible post-August input acquisition, external publication-time verification, and reproduction of technical indicators on new data remain shared/open implementation issues; the five-day staleness rule is fixed. If the CV-selected E2 configuration needs features beyond F0, resolve that conflict before deployment; do not silently drop those features. No frozen-model training or daily inference code is implemented by this strategy document.

## 5. Exact F0 baseline: 21 logical feature fields — LOCKED

Conventions: `open`, `close`, `high`, `low` are the **adjusted** stock prices. `t-1`, `t-2`, etc. refer to preceding **per-stock trading sessions** under the central date policy; actual publication times remain unverified. For a 20-session window, use exactly `t-20 ... t-1` inclusive. Compute causal lag formulas from original stock time series before slicing into folds; fitted transformations still use fold training data alone. Undefined ratios become missing without dropping rows. For globals, "latest" means the latest nonmissing observation dated strictly before `t`, no more than five calendar days old for that series.

| # | Baseline name | Source / calculation |
|---:|---|---|
| 1 | `intraday_return_1d` | `close[t-1] / open[t-1] - 1` |
| 2 | `daily_return_1d` | `close[t-1] / close[t-2] - 1` |
| 3 | `momentum_5d` | `close[t-1] / close[t-6] - 1` |
| 4 | `price_range_1d` | `(high[t-1] - low[t-1]) / close[t-1]` |
| 5 | `relative_volume_20d` | `volume[t-1] / mean(volume[t-20:t-1])` |
| 6 | `relative_amount_20d` | `amount[t-1] / mean(amount[t-20:t-1])` |
| 7 | `trade_size_ratio_20d` | `(volume/trade_count)[t-1] / mean((volume/trade_count)[t-20:t-1])` |
| 8 | `foreign_ownership` | `foreign_ownership_percent[t-2] / 100` |
| 9 | `foreign_net_buy_ratio` | `foreign_net_buy_shares[t-2] / volume[t-2]` |
| 10 | `trust_net_buy_ratio` | `trust_net_buy_shares[t-2] / volume[t-2]` |
| 11 | `rsi_normalized` | `RSI[t-1] / 100` |
| 12 | `sma_distance` | `close[t-1] / SMA[t-1] - 1` |
| 13 | `macd_normalized` | `MACD[t-1] / close[t-1]` |
| 14 | `atr_normalized` | `ATR[t-1] / close[t-1]` |
| 15 | `bb_width` | `(BBANDS_upper[t-1] - BBANDS_lower[t-1]) / BBANDS_mid[t-1]` |
| 16 | `vix_normalized` | `latest_eligible(^VIX) / 100` |
| 17 | `nasdaq_return` | `latest_eligible(^IXIC) / preceding_observed(^IXIC) - 1` |
| 18 | `sp500_return` | `latest_eligible(^GSPC) / preceding_observed(^GSPC) - 1` |
| 19 | `taiex_return` | `latest_eligible(^TWII) / preceding_observed(^TWII) - 1` |
| 20 | `stock_id` | Categorical string stock ID from file/metadata, **not an ordinal numeric variable** |
| 21 | `day_of_week` | Prediction trading date weekday, categorical |

Raw source-column mapping for formulas: `close`=`etl:adj_close`; `open`=`etl:adj_open`; `high`=`etl:adj_high`; `low`=`etl:adj_low`; `trade_count`=`price:成交筆數`; `amount`=`price:成交金額`; `volume`=`price:成交股數`; `foreign_ownership_percent`=`foreign_investors_shareholding:全體外資及陸資持股比率`; `foreign_net_buy_shares`=`institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)`; `trust_net_buy_shares`=`institutional_investors_trading_summary:投信買賣超股數`.

**F0 = 21 logical fields**, not necessarily 21 numeric encoded columns; `stock_id` and `day_of_week` expand under one-hot encoding. F0 includes normalized/relative financial values and two categorical context fields. No additional raw unnormalized price/volume predictors in E1. Raw originals remain available to members' E2 experiments.

**Shared dataset interface:** `uv run python -m shared.features --data-dir data --output data/processed/f0.csv` rebuilds F0 from the unchanged raw files. `load_f0_dataset("data/processed/f0.csv")` returns all **161,746** stock/date rows with `date` plus exactly the 21 logical F0 fields. The generated file is gitignored and contains no `y`, target-day OHLC, or realized return. Members select fold rows by immutable keys, obtain training labels and evaluation prices separately from `load_market`, and fit `BaselinePreprocessor` on each training fold only. All four E1 approaches must use this one shared feature definition and alignment policy.

## 6. Baseline preprocessing: SAME for all algorithms — LOCKED

The following applies to **E1** across all four methods:

1. Detect division by zero and undefined numerical calculations; convert `+inf` and `-inf` to `NaN`. Do not silently drop stock-date observations.
2. Numeric F0 columns: **median imputation fitted using training rows only**, followed by **StandardScaler fitted on training rows only**. Apply the fitted objects to validation/test.
3. Categorical F0 columns: replace feature missingness with a literal `Unknown`, then apply a **shared one-hot encoding policy** learned from training rows. Handle unseen categories deterministically (`handle_unknown='ignore'` is a viable implementation).
4. **No baseline winsorization, clipping, outlier removal or feature selection.** Model-specific preprocessing choices belong in **E2**, not E1.
5. Keep `stock_id` and `date` as immutable row keys, even if `stock_id` also enters F0 as a categorical predictor. A missing key is a **data integrity error**; do not fill a missing key with `Unknown`. Likewise, missing historical training labels or evaluation realized returns require explicit handling/diagnostics, not silent relabeling.
6. Strictly fit one preprocessing instance **per training fold**. For final holdout refit on all 2020–2025 training rows. Preserve the exact 50-stock validation/test keys for every model.

The implemented baseline raises on an all-missing numeric feature within a training fold. The global maximum age is fixed at five calendar days per series; older observations become missing. Technical-indicator replay on newly acquired data remains unresolved. Do not silently substitute invented values.

**Classifier threshold:** initial convention is `p(up) > 0.5 -> 1`, otherwise `0`. Decision threshold optimization is allowed in E2/CV only and must not use holdout labels. Do not assume all algorithms must expose calibrated probabilities.

## 7. Recommended scaffold and file contracts

The original proposed skeleton was (see the root README for implemented files):

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
pyproject.toml
uv.lock (generated by uv when dependencies are resolved)
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
7. Test strict-prior-date global joins, five-calendar-day staleness, independent series, and missing values. Separate publication-time verification remains unresolved; synthetic date-alignment tests do not certify real publication times.

## 8. Submission and phase-specific integrity

Generate both phases from their **actual phase/session-specific** `sample_submission.csv` templates, preserving header spelling, stock ID order and exact dates. Validate every prediction is `0` or `1`; reject missing/duplicated/reordered/additional stock IDs. Phase 1: **4,200 predictions**. Phase 2: **50 predictions**. Daily Phase 2 only counts the latest valid file submitted before 09:00 Taiwan time. Confirm upload in Submission Status.

Archive supplied data only through **2026-08-31**: obtaining reproducible **September–December live inputs**, determining the eligible deployment-refit window, and integrating the frozen-artifact daily inference workflow are **SHARED / OPEN operational tasks**. Daily Phase 2 inference does not retrain. Do not claim trained or deployed Phase 2 readiness while these are unresolved.

## 9. Ownership matrix — do not confuse these

| Shared/fixed | Independent/member-owned | Shared/open to verify |
|---|---|---|
| Target and official scorer | Model selection after team assignment | External publication-time verification |
| 100-stock pooled E1 train and 50-stock eval | Hyperparameter optimizer and tuning budget | Publication time of institutional features |
| Same chronological folds and test | E2 feature engineering and preprocessing | Technical-indicator reproducibility |
| E1 seeds `42`, `64`, `128`, `256`, `512` and five-day global maximum age | E2 feature selection/ablation | Actual source publication delays |
| Baseline 21 F0 fields and shared E1 preprocessing | Optional 100-vs-50 training ablation | Results schema implementation and secondary metrics |
| Historical pre-09:00 cutoff, test isolation, and frozen daily inference | Model-specific diagnostics | Live input acquisition, deployment-refit window, artifact storage, submission integration |

If a user later changes a **LOCKED** item, update `context/README.md` and `AGENTS.md` together, preserve the old protocol record, and record the revision in a new protocol document. Keep the root `README.md` aligned with implemented behavior. If only a member-owned experiment changes, do not modify the central baseline contract.

**Avoid making unsupported claims about model superiority, leakage-free data, up-to-date global timestamps, or competition performance. Report what has actually been verified.**
