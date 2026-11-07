# NTU Machine Learning Final Project — Shared Experiment Protocol

**Course:** NTU Machine Learning, Fall 2026 (instructor: Hsuan-Tien Lin)  
**Protocol:** Team-agreed v1.1, updated 2026-11-07  
**Status:** Ready to share and scaffold. **This document describes a plan; it does not claim the repository code or experiments already exist.**

> **Decision labels:** **LOCKED** means the team has agreed. **MEMBER-OWNED** means the individual approach owner may decide and must document the choice. **SHARED / OPEN** means the team (or a designated shared-infrastructure owner) must verify or settle it before declaring the implementation scientifically valid. The official course PDF takes precedence over this plan.

## 1. Project goal and constraints

Predict the **intraday direction** of a fixed set of **50 Taiwanese stocks**. For stock `i` on trading date `t`:

```text
y[i,t] = 1  if adjusted_close[i,t] > adjusted_open[i,t]
y[i,t] = 0  otherwise (including no change)

r[i,t] = (adjusted_close[i,t] - adjusted_open[i,t]) / adjusted_open[i,t]
official_score = mean((2 * predicted_y[i,t] - 1) * r[i,t])
```

Higher official score is better. Ordinary classification accuracy is **not** the competition's scoring objective. The required survey report studies **at least four meaningfully different machine learning approaches**, compares them from multiple perspectives, and recommends **one best reproducible approach**, explaining its advantages and disadvantages.

| Deliverable | Points / share of project | Relevant deadline (Asia/Taipei, UTC+8) |
|---|---:|---|
| Phase 1 static CSV | 120 / 1,920 (6.25%) | **2026-12-24 23:59**; 50 stocks × 84 dates (2026-06-01 through 2026-09-30), 4,200 predictions; best accepted version counts |
| Phase 2 competition sessions | 360 / 1,920 (18.75%) | Each session closes at **09:00 on its trading date**; 50 predictions/session; latest valid submission counts |
| Final survey report | 1,440 / 1,920 (75%) | **2026-12-30 13:00**, one report/team via Gradescope, maximum seven A4 pages |

Phase 2 graded dates are 2026-11-16–20, 11-23–27, 11-30–12-04, 12-07–11, 12-14–18 and 12-21–24. Ordinary sessions before 2026-11-15 do not contribute to the grade. Missing a valid graded-session submission generates an **all-zero prediction file**, which does not necessarily score numerically zero. Keep source code through **2027-01-31 23:59** for possible inspection. Follow all honesty and citation rules in the official specification.

**Authoritative inputs (supplied separately):** `final_project.pdf`, `fin_pred_data.zip`, and phase/session-specific `sample_submission.csv` from <https://final.html2026.csie.org>.

## 2. Team and ownership

| Member | Primary method assignment |
|---|---|
| 鄭振慈 | To be decided |
| 葉家堯 | To be decided |
| 龔建瑄 | To be decided |
| 倪永珅 | To be decided |

**LOCKED:** four members, **one primary ML approach per member**, personal experiment folders, and shared data/evaluation infrastructure. Candidate families *under discussion* are Logistic Regression, Random Forest, LightGBM/XGBoost (as a boosted-tree research track), and a multilayer perceptron. One member expressed interest in LightGBM/XGBoost, but **no names-to-models assignment is finalized**.

### Proposed repository layout for Codex to create

```text
ML-Final-Project/
├── README.md                         # This shared human-readable protocol
├── AGENTS.md                         # Coding-agent context and guardrails
├── data/
│   ├── raw/                          # Original extracted data; immutable, gitignored
│   ├── processed/                    # Rebuildable data artifacts; gitignored
│   └── splits/                       # Versioned manifests / split settings
├── shared/
│   ├── data_loader.py
│   ├── features.py                   # Common 21-feature F0 definitions
│   ├── preprocessing.py              # Common baseline preprocessing
│   ├── splits.py                     # Chronological fold logic
│   ├── evaluation.py                 # Official scorer and validation checks
│   └── submission.py                 # CSV validation; implementation later
├── members/
│   ├── 鄭振慈/
│   ├── 葉家堯/
│   ├── 龔建瑄/
│   └── 倪永珅/
├── results/                           # Shared result schema; outputs usually gitignored
├── submissions/                       # Generated CSVs; preserve submission templates
├── tests/                             # Shared integrity, split and leakage tests
├── configs/                           # Shared experiment, feature and fold configuration
├── requirements.txt                   # To be populated after tools/deps are chosen
└── .gitignore
```

The structure is a **proposal for scaffolding**, not a claim of existing implementation. UTF-8 Chinese member folder names are acceptable; use safe imports and paths. Keep raw archives, large derived files, credentials and submission secrets out of Git. Version scripts, configs, schema and reproducible split definitions instead.

## 3. Verified data inventory

The supplied `fin_pred_data.zip` has:

- `data/market/<stock_id>.csv`: **100 stock files**, **161,746** stock-date rows, covering **2020-01-02 through 2026-08-31**. Each includes `date`, **19 stock-level numerical predictor columns**, and label `y`.
- `data/global/global_data.csv`: `date`, `^VIX`, `^IXIC`, `^GSPC`, `^TWII`, `^KS11`, `TX_OI` — **six global series**.
- `data/info.json`: `Top50@Y` (the **50 target stock IDs**) and `Top100@Y` (100 provided stocks).
- Only **eight stock-predictor missing values** were found in the supplied stock records; some stock trading dates lack corresponding same-date global rows. **Same-date coverage is not proof of data availability.**

The label `y` in the supplied files matches `adjusted_close > adjusted_open`. **Never include `y` or the target-day adjusted OHLC / realized intraday return in model input.** The provided labels extend through August 2026, overlapping part of Phase 1; that portion is **not** an independent holdout if used for training or tuning.

See `AGENTS.md` for the full raw schema, mappings and precise F0 formulas.

## 4. LOCKED: shared experimental protocol

### 4.1 One stock-date = one sample

- Rows are keyed by `(stock_id, date)`, with stable string IDs and consistent date parsing.
- The initial baseline is a **pooled model trained on all 100 supplied stocks**.
- **Validation and final testing evaluate only the 50 target stocks** from `Top50@Y`.
- All approaches must predict the **exact same** target-stock/date keys for each evaluation fold. A member may not drop evaluation rows because of a feature-imputation issue.
- **MEMBER-OWNED additional experiment:** compare training on all 100 versus only the 50 target stocks. Keep evaluation universe and folds unchanged; hold model setup fixed for an initial controlled ablation. The result is unknown until measured.

### 4.2 Historical prediction information cutoff

**LOCKED:** Simulate a prediction cutoff of **strictly before 09:00 Asia/Taipei on prediction date `t`** for historical data and Phase 1, matching Phase 2's operational requirement.

- Use prior **completed and published** stock sessions for prices, volume, technical features and institutional data.
- Never use target-day intraday or end-of-day information at a pre-market cutoff.
- Join global features using **actual availability timestamps**, not calendar dates alone; handle time zones, US overnight markets, Taiwan market close, holidays and source delays.
- **SHARED / OPEN BLOCKER:** raw global data contains calendar dates but no verified per-series publication timestamp. Institutional series availability and supplied technical-indicator reproduction also need checking. Do **not** silently assume data was observable just because a historical value is in a CSV; flag an unresolved source or use a documented verified-safe lag.

### 4.3 Three expanding-window folds + independent holdout

| Run | Training window (100 stocks) | Validation/test window (50 target stocks) |
|---|---|---|
| Fold 1 | All available dates in **2020–2022** | **2023** |
| Fold 2 | All available dates in **2020–2023** | **2024** |
| Fold 3 | All available dates in **2020–2024** | **2025** |
| Final holdout | Refit using **2020–2025** | **2026-01-01 through 2026-05-31** |

- Split by **trading date**, never randomly by row.
- Use the **single team seed `42`**. Extra seeds are optional diagnostics, not required for the primary study.
- Fit feature imputers, encoders and scalers **separately on each training fold**, then transform validation with the fitted objects. Refit preprocessing on 2020–2025 for final holdout.
- **Do not tune on the Jan–May 2026 test.** Evaluate once after freezing the configuration. Do not choose a lucky seed, model or extra feature after seeing final test results.

### 4.4 Selection criterion

**LOCKED:** select hyperparameters and feature configurations using the **equal-weight mean of the three yearly validation official scores**, not mean accuracy:

```text
CV_score(config) = [official_score_2023(config)
                  + official_score_2024(config)
                  + official_score_2025(config)] / 3
best_config = argmax(CV_score)
```

Each fold's official score averages its **50 target-stock predictions over the validation dates actually present**. Record all three fold scores; don't inadvertently weight folds according to sample counts. Accuracy may be reported as a supplementary metric; the full set of secondary report metrics is still open.

### 4.5 Two experiments per approach

- **E1 — Shared baseline:** every member uses the **same 21 logical baseline features F0**, shared baseline preprocessing, data partitions and official scorer. This controls the input data and preprocessing while comparing algorithms.
- **E2 — Member-optimized pipeline:** each member can independently engineer further *causally available* features, select features, adjust preprocessing and tune the model, while preserving the shared evaluation keys, cutoff, folds and scorer. Compare E2 to E1, and compare all four final E2 pipelines.

**MEMBER-OWNED:** hyperparameter tuning technique (grid/random/Bayesian/Optuna/etc.), ranges, number of trials, model-specific enhancements and extra ablations. Record tuning budget (trials, compute time) because different budgets affect comparison fairness. Members should not quietly redefine F0 or the common baseline to improve a score.

## 5. LOCKED: baseline F0 (21 logical features)

We selected **transformed/relative financial variables**, not raw stock price and volume levels, along with two categorical context fields. Each numeric feature uses information known at the cutoff; `t-1` below is the prior **stock trading session**, not necessarily yesterday's calendar date.

| Group | Shared feature names | Count |
|---|---|---:|
| Price/momentum | `intraday_return_1d`, `daily_return_1d`, `momentum_5d`, `price_range_1d` | 4 |
| Trading | `relative_volume_20d`, `relative_amount_20d`, `trade_size_ratio_20d` | 3 |
| Institutional | `foreign_ownership`, `foreign_net_buy_ratio`, `trust_net_buy_ratio` | 3 |
| Technical | `rsi_normalized`, `sma_distance`, `macd_normalized`, `atr_normalized`, `bb_width` | 5 |
| Global | `vix_normalized`, `nasdaq_return`, `sp500_return`, `taiex_return` | 4 |
| Context | `stock_id`, `day_of_week` | 2 |
| **Total** | | **21** |

These are **21 logical inputs**, **not** necessarily 21 encoded numerical columns: one-hot encoding `stock_id` and weekday will increase the matrix width. The detailed formulas, 20-session rolling window conventions and raw-source mappings are in `AGENTS.md`. Some timestamp/indicator conventions are open technical verification items; the selection of the 21 features is locked.

## 6. LOCKED: common E1 preprocessing

**Apply the following single common baseline pipeline to all four approaches** (members can change it in E2 only):

1. Convert invalid numerical values (`+inf`, `-inf`, e.g. division by zero) to missing (`NaN`). Do not silently drop samples.
2. **Numerical:** training-fold **median imputation**, then training-fold **StandardScaler** for numerical F0 fields.
3. **Categorical (`stock_id`, `day_of_week`):** fill missing feature categories with `Unknown`, then **one-hot encode**, learning the vocabulary from the training fold. Handle unseen categories deterministically without leaking future categories.
4. **No default outlier clipping, winsorization or feature selection** in E1.
5. Apply the exact fitted preprocessing instance to the corresponding validation/test set. Never fit it on validation/test rows or on all dates combined.
6. Preserve every agreed evaluation key. Missing `stock_id`/`date` identifiers and missing required evaluation labels are **integrity errors**, not ordinary categories to impute.

**SHARED / OPEN edge cases:** decide explicitly how to handle an all-missing numeric feature within a particular training fold; establish safe rules for stale global observations and unavailable indicator values. Implement fail-fast errors or explicit logged policy rather than inventing replacements. For probability-producing classifiers, `p(up) > 0.5 -> 1` is the initial convention; custom thresholds may be tuned on CV folds only.

## 7. What belongs to the individual approach owner?

Each member owns their assigned model code and can make method-specific choices, provided they are **reproducible** and obey all locked constraints:

| Member-controlled | Examples |
|---|---|
| Primary method and architecture (once assigned) | LR, RF, boosted trees, MLP |
| Hyperparameter search | Search method, ranges, resource budget |
| E2 feature engineering | Lag windows, feature interactions, market features, feature selection |
| E2 preprocessing | Scaling alternatives, category encoding, outlier treatment |
| E2 additional experiments | 100-vs-50-stock ablation; return regression vs classification; other hypotheses |
| Diagnostics | Feature importance, interpretability, computational efficiency |

**Shared decisions that members should not fork:** stock/date keys, target definition, F0 baseline definitions, pre-09:00 availability rules, common E1 preprocessing, chronological folds, official score implementation and evaluation format.

## 8. Standardized experiment outputs (recommended scaffold contract)

Create a common results format so everyone can compare runs independently of their model implementation:

- `metrics.csv`: at least `member`, `model`, `experiment` (`E1`/`E2`), `config_id`, `fold`, `seed`, `official_score`, `accuracy`, `n_eval`, `train_seconds`, `predict_seconds` (measurement policy to be documented).
- `predictions.csv`: `member`, `model`, `experiment`, `config_id`, `fold`, `stock_id`, `date`, `predicted_y`; optionally `p_up` when available. Keep true labels / realized returns in the shared evaluation dataset rather than inventing them for competition rows.
- `config.json` or `config.yaml`: model parameters, feature list/version, preprocessing variation, optimization method, trial count, seed, package versions and data/split version.

These file names are **recommended repository interfaces**, not officially prescribed report or submission formats. Always validate one prediction per required evaluation stock-date key, values in `{0,1}`, and no duplicates, missing IDs or reordering in competition CSVs.

## 9. Submission and live-data work: still open

- Phase 1 and Phase 2 submissions must be filled from the appropriate **actual** `sample_submission.csv`. Do not invent the date headers or stock order. Phase 1 has 4,200 binary cells; Phase 2 has 50 binary cells for the session date.
- Check Submission Status after upload; a `Pending` Phase 2 submission is accepted but awaiting scoring. Only the latest valid Phase 2 version counts.
- The provided ZIP stops at **2026-08-31**. Future deployment needs **reproducible post-August market data**, verifiable publication times, an update process and a model-retraining policy. **These are team-level operational decisions, not safe individual shortcuts.**

## 10. What remains to decide?

**SHARED / OPEN (must be coordinated):** authoritative global and institutional timestamps; reproducibility/availability of supplied indicators; zero-denominator/all-missing/stale-input safeguards; final secondary metric definitions; result-schema implementation; Phase 2 data updating and submission/retraining orchestration; specific model/member assignments.

**MEMBER-OWNED:** choice of hyperparameter search algorithm/ranges/trials; optional extra features and features-selection methods in E2; optimized preprocessing; optional independent ablations including 100-versus-50 training stocks; algorithm-specific diagnostics.

**Next step for Codex:** create the agreed repository skeleton and shared interfaces/tests first. **Do not silently fabricate timestamp metadata, download private data, train models, or turn open decisions into assumptions.** Read `AGENTS.md` before implementation.
