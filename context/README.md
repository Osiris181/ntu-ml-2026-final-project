# Team guide

We are building four different machine learning approaches to predict whether a Taiwanese stock closes above its adjusted opening price on a trading day. Each prediction is `0` or `1` for one `(stock_id, date)` pair. The competition score rewards correct **direction weighted by that day's intraday return**, so we choose models using the official score.

```text
official_score = average((2 × prediction - 1) × (adjusted_close - adjusted_open) / adjusted_open)
```

The [main README](../README.md) has the exact F0 formulas and public project overview. The [original protocol](protocol-v1.1.md) keeps the detailed team decisions; the [course PDF](../final_project.pdf) is authoritative.

## Files in this directory

| File | Use |
|---|---|
| [`README.md`](README.md) | This practical guide for members: available inputs, E1/E2 work, shared pipeline, and model inputs/outputs. |
| [`protocol-v1.1.md`](protocol-v1.1.md) | Original protocol record. Use the current root README and `AGENTS.md` for later agreed changes. |

## What we have

| Part | What it contains |
|---|---|
| Stock data | 100 CSV files in `data/market/`, covering 2020-01-02 to 2026-08-31. Each row is one stock and trading date, with market fields and the historical label `y`. |
| Global data | `data/global/global_data.csv` with VIX and market index series. |
| Stock lists | `data/info.json`: `Top100@Y` lists the 100 supplied stocks; `Top50@Y` lists the 50 stocks we must evaluate. |
| Shared code | `shared/` builds the common F0 dataset, defines date splits and E1 preprocessing, scores predictions, and exports submission files. |
| Shared F0 file | `data/processed/f0.csv` has one feature row for every supplied stock/date row. Regenerate it from raw data when needed; it is gitignored. |
| Personal workspaces | One folder per member in `members/`. No models or results have been added yet. |

The label `y` says whether adjusted close was above adjusted open on that date. It is used for training and scoring, never as a model input. The target day's prices, volume, and indicators are also excluded from input features. Our cutoff is **strictly before 09:00 Asia/Taipei on the prediction date**; the shared date-lag rules below are the team's conservative, unverified proxy for availability.

## Our shared F0 features

**F0** is the common set of 21 input fields used in E1. For stock data, `t-1` means the **previous trading session for that stock**, which may be several calendar days earlier; institutional features use `t-2`. A 20-session average uses exactly the previous 20 stock trading sessions. These are mostly relative values, so E1 does not feed raw stock price or volume levels into a model.

| Group | Feature names | What they tell the model |
|---|---|---|
| Price movement (4) | `intraday_return_1d`, `daily_return_1d`, `momentum_5d`, `price_range_1d` | Previous-session open-to-close change, close-to-close change, five-session momentum, and high-to-low range. |
| Trading activity (3) | `relative_volume_20d`, `relative_amount_20d`, `trade_size_ratio_20d` | Previous-session volume, traded amount, and average trade size compared with their 20-session averages. |
| Institutional activity (3) | `foreign_ownership`, `foreign_net_buy_ratio`, `trust_net_buy_ratio` | Foreign ownership and foreign/trust net buying from `t-2`, divided by `t-2` volume where applicable. |
| Technical indicators (5) | `rsi_normalized`, `sma_distance`, `macd_normalized`, `atr_normalized`, `bb_width` | Normalized RSI, distance from moving average, normalized MACD and ATR, and Bollinger Band width. |
| Global markets (4) | `vix_normalized`, `nasdaq_return`, `sp500_return`, `taiex_return` | Each series' latest nonmissing observation dated before prediction day, at most five calendar days old. |
| Context (2) | `stock_id`, `day_of_week` | Which stock and which weekday is being predicted. Both are categorical. |

That is **21 logical fields**. One-hot encoding expands the two categorical fields into more numeric columns. The exact formulas and raw CSV column mappings are in the [main README](../README.md) and [protocol](protocol-v1.1.md).

## Approaches

The team will compare **four meaningfully different approaches**, one primary approach per member. The candidate families discussed so far are Logistic Regression, Random Forest, boosted trees (LightGBM or XGBoost), and a multilayer perceptron. **These are candidates; the model-to-member assignments are not finalized.**

Each member runs two versions of their approach:

- **E1, shared baseline:** compare the four algorithms under the same conditions. Everyone trains on all 100 stocks, uses only the 21 F0 fields, uses the same preprocessing, predicts the same 50 target-stock rows, and uses the same folds, seed `42`, and scorer. The member implements their own algorithm; the shared parts stay fixed.
- **E2, your improved version:** use E1 as your comparison point, then try improvements you can explain and reproduce. You may add features known before the cutoff, change preprocessing, and tune your algorithm. Keep the target rows, cutoff, folds, and official scorer unchanged. Compare E2 with your own E1.

For E1, the shared preprocessor replaces invalid numeric values with missing values, fills missing numeric values with **medians learned from the training fold**, then standardizes numeric columns using that fold. It fills missing feature categories with `Unknown` and one-hot encodes categories learned from the training fold. It has no default clipping or feature selection. Fit a **new** preprocessor for each fold; transform validation rows with that fitted instance.

## Shared pipeline

```text
Supplied CSVs + stock lists
          ↓
Load and validate rows
          ↓
Build the shared F0 dataset with fixed t-1/t-2 and global date rules
          ↓
Split by date: train on 100 stocks, evaluate on the same 50 target stocks
          ↓
Fit preprocessing on training rows only
          ↓
Member trains their approach and predicts evaluation rows
          ↓
Shared code checks predictions and calculates scores
```

We use three expanding validation folds: **2020–2022 → 2023**, **2020–2023 → 2024**, and **2020–2024 → 2025**. The team's seed is `42`. Choose a configuration using the **average of the three yearly official scores**, giving each year equal weight. The **Jan–May 2026 holdout** is used only after model choices are fixed.

Members load the same F0 file directly; no member needs to implement their own historical lag rules:

```text
python -m shared.features --data-dir data --output data/processed/f0.csv
```

```python
from shared.features import load_f0_dataset

f0 = load_f0_dataset("data/processed/f0.csv")
```

`f0` contains feature columns and stock/date keys only. Join by those keys to the raw market rows for training labels and evaluation prices. Apply each fold's fitted preprocessing only after selecting that fold's rows.

### Use the shared F0 file in E1

Run the F0 generation command above if `data/processed/f0.csv` is absent. Then, from the repository root, prepare the first validation fold like this:

```python
from shared.data_loader import load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, split_market

target_ids, _ = load_metadata("data")
market = load_market("data")
f0 = load_f0_dataset("data/processed/f0.csv")

train_rows, valid_rows = split_market(market, target_ids, FOLDS[0])
keys = ["stock_id", "date"]
train = train_rows[keys + ["y"]].merge(
    f0, on=keys, how="left", validate="one_to_one", indicator=True
)
valid = valid_rows[keys].merge(
    f0, on=keys, how="left", validate="one_to_one", indicator=True
)
if not train["_merge"].eq("both").all() or not valid["_merge"].eq("both").all():
    raise ValueError("A split row is missing from the shared F0 file")

preprocessor = BaselinePreprocessor()
X_train = preprocessor.fit_transform(train[list(F0_FEATURES)])
X_valid = preprocessor.transform(valid[list(F0_FEATURES)])
y_train = train["y"]
valid_keys = valid[keys]
```

`X_train` and `y_train` go to your member model; `X_valid` is its validation input. Keep `valid_keys` with the model's predictions so the shared scorer can match every target stock/date row to `valid_rows`. The F0 CSV intentionally keeps some missing feature values: `fit_transform` learns numeric medians and scaling **only from this training fold**, while `transform` applies those fitted values to validation. Never fit the preprocessor on `valid` or on all of `f0`.

Repeat with `FOLDS[1]` and `FOLDS[2]`, creating a **new** `BaselinePreprocessor` and model for each fold. The folds train on all 100 stocks and validate on the same 50 target stocks. The shared scorer checks that no required prediction is missing and calculates each fold's score.

After your member-owned model has produced binary predictions for `X_valid`, attach them to the matching keys and call the shared scorer:

```python
from shared.evaluation import evaluate_predictions

predictions = valid_keys.copy()
predictions["predicted_y"] = model.predict(X_valid)  # your fitted member model
fold_result = evaluate_predictions(valid_rows, predictions)
```

Record `fold_result["official_score"]` separately for each of `2023`, `2024`, and `2025`. Once all three folds are done, use `cv_score` from `shared.evaluation` on a dictionary of those yearly scores to compare configurations. The [shared module guide](../shared/README.md#what-members-call-in-e1) lists every common call, its input, and its output. The Jan–May 2026 `HOLDOUT` is for evaluation only after model selection.

## What goes into and comes out of a member's model

| Item | Meaning |
|---|---|
| Training input | F0 features for training rows, prepared by a new preprocessor fitted on that fold's training rows only. |
| Training target | The historical binary `y` for those rows. `y` is never an input feature. |
| Evaluation input | The same F0 columns for the 50 target-stock rows in the evaluation year, transformed by the fitted preprocessor. |
| Required output | One `predicted_y` value, `0` or `1`, for **every** required `(stock_id, date)` key. Do not drop rows. |
| Recorded result | Official score, accuracy, number of evaluated rows, model settings, seed, and the predictions for each fold. |

Use `shared/splits.py` for the folds, `shared/preprocessing.py` for E1, and `shared/evaluation.py` for key checks and scoring. Keep model code in your own `members/<name>/` folder. Record what changed between E1 and E2 and how much tuning you performed.

## Turning predictions into a submission CSV

The supplied `sample_submission.csv` has 50 stock rows and 84 date columns for Phase 1. Members should keep model output in **long form**: one row with `stock_id`, `date`, and `predicted_y` for each required prediction. The export utility converts it to the template's **wide form**: `id` followed by the template's date columns, in exactly the template's order.

```python
from shared.data_loader import load_metadata
from shared.submission import write_submission

target_ids, _ = load_metadata("data")
write_submission(
    predictions,  # DataFrame: stock_id, date, predicted_y
    "sample_submission.csv",
    "submissions/phase1.csv",
    target_ids,
)
```

The utility requires all 4,200 Phase 1 cells to be supplied by the caller; it does not fill gaps with the template's zeros. It rejects extra or duplicate keys and nonbinary values, preserves the template header and stock order, and refuses to overwrite the template or an existing output file. For a later session, use that session's **actual** template in the same call.

## What needs team agreement first

The stock CSVs **already include** RSI, SMA, MACD, ATR, and Bollinger Band values. F0 uses those values from prior stock sessions; we do not need to recalculate them for the supplied historical data. Their calculation settings would matter if we later need to reproduce them for new data.

The shared F0 file is generated under the team's conservative date policy. The CSV dates do **not** prove when market, institutional, technical, or global values were actually published; the five-day global limit and stock-session lags are not externally verified availability timestamps. The supplied technical-indicator settings also remain undocumented for reproduction on new data. See the central **Data quality and timing assumptions** section in the [main README](../README.md). Do not claim the date policy has been independently certified as leakage-free.

The team also needs to finalize model assignments. Later, live competition work will need updated post-August data and the actual session template. Submission-file export is implemented, but live-data updates, trained predictions, and uploading are not.

## Quick check

From the repository root, install `requirements.txt` and run:

```text
python -m unittest discover -s tests -v
```
