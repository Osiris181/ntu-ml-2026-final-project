# E9 public investor-positioning data: feasibility gate

**Decision: stop before E9 implementation.** Public exchange reports appear to cover the required calendar years in principle, but this review could not verify a reproducible, complete stock-date extract or its point-in-time availability for our 2020–2025 training and validation rows. E3 remains the reference approach. No E9 model was trained or evaluated.

## Sources and coverage checked

| Source | What it establishes | What remains unverified |
|---|---|---|
| [TWSE margin-balance query](https://www.twse.com.tw/zh/trading/margin/mi-margn.html) and [stock-level report](https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=20260814&response=html&selectType=ALL) | TWSE advertises dated CSV queries from January 2001 and shows per-security financing and short-selling purchases, sales, and balances. | A working, repeatable extraction of every required 2020–2025 date; coverage of each target and training stock on those dates. |
| [TPEx margin-balance query](https://www.tpex.org.tw/zh-tw/mainboard/trading/margin-trading/transactions.html) and [government dataset description](https://data.gov.tw/dataset/11387) | TPEx advertises dated reports from January 2007. The government catalog lists date, stock code, prior/current financing and short balances, transactions, limits, and remarks. | Date-by-date historical extraction, exact historical coverage and units, and missing-row semantics. |
| [TWSE regulation, Article 7](https://twse-regulation.twse.com.tw/TW/law/DAT0201.aspx?FLCODE=FL007018) | Reporting institutions send balances each afternoon; exchanges announce them before the **next** business day's market opening. | An exact historical publication timestamp before our stricter **09:00 Asia/Taipei** cutoff for every stock-date record. |

The project metadata has **100 training stock IDs and 50 target IDs**. At least two target IDs, [8069](https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=htm) and [3293](https://www.tpex.org.tw/zh-tw/mainboard/listed/latest/detail.html?code=3293), are TPEx/OTC securities, so TWSE data alone would not cover the evaluation universe. The official start dates support possible 2020–2025 calendar coverage, **not** verified coverage for our individual stocks or sessions. There are no margin or short-balance files in the local raw-data folders. A direct request for the TWSE 2020-01-02 report failed to connect from this workspace; TPEx pages returned access errors in the available web client. Those access failures do not prove the public sites are generally unavailable, but they prevent a reproducible local coverage audit now.

## Definitions, missingness, timing, and revisions

The exchange reports distinguish **financing/margin balances and transactions** from **short-sale balances and transactions**. They also have prior-day and current-day balances, which must not be mixed or assumed interchangeable. The [TWSE explanation](https://www.twse.com.tw/exchangeReport/MI_MARGN?response=html&selectType=MS) says its “today” balance can be revised as institutions reconcile the next day, and directs readers to the subsequent report's “previous balance” as authoritative. Thus a modern historical download may contain corrected figures; without archived versions or a verified revision policy, it cannot automatically be treated as the exact value visible at the time. The [TPEx catalog](https://data.gov.tw/dataset/11387) includes remarks and eligibility information, but this review could not measure missing stock-days. An absent row may mean ineligibility, suspension, missing publication, or a source/identifier issue; it must remain missing until classified. A recorded zero balance is different from an absent row.

For a prediction on trading day `t`, **proposed conservative alignment** is to use only the finalized balance for stock session `t-2`, as represented in the report issued for `t-1`, and to derive changes only from `t-2` and `t-3` balances. This is a proposed safeguard based on the exchange's next-day reporting/revision description; it is **not proof** that each value or corrected historical version was available before 09:00 on `t`. Verify actual report timestamps and how later corrections are reflected before adopting this rule. Never use target-day or `t-1` current-day figures merely because a historical endpoint returns them today.

F0 already includes lagged stock price, volume, foreign ownership/net buying, trust net buying, and global market context. It contains **no per-stock financing or short-sale balances**; those fields would add a distinct positioning signal. Do not confuse short-sale financing records with the existing institutional net-buy ratios.

## Conditional E9 design after the data gate passes

If complete official TWSE **and** TPEx snapshots can be acquired and audited, predefine only these three numeric candidate features in `members/zhenci/e9/`:

1. `margin_balance_change`: `(margin_balance[t-2] - margin_balance[t-3]) / (1 + margin_balance[t-2] + margin_balance[t-3])`.
2. `short_balance_change`: the same bounded change for short-sale balance.
3. `short_share_of_credit_balance`: `short_balance[t-2] / (margin_balance[t-2] + short_balance[t-2])`, missing when both balances are zero or either is unavailable.

Confirm a common stock-unit convention for TWSE and TPEx before calculating these. Keep missing values as missing and use the **same fold-fitted numeric median imputation and scaling** as the shared baseline; do not alter shared F0. Train E9 with the **existing E3 XGBoost configuration** on the same 100-stock expanding training windows and evaluate on the same 50 target-stock keys, applying the fixed E3 TAIEX decision rule and the official score. Compare E3, E9, and the separate previous-direction candidate by validation year, equal-year mean, accuracy, predicted-up fraction, per-feature missingness, and prediction/score distributions. Do not use Phase 1, Jan–May 2026, or the October check for feature or parameter selection. This is a design only; no E9 feature or model code exists yet.

## Minimum evidence to reopen the gate

Acquire and preserve dated TWSE and TPEx response files for the required 2020–2025 sessions, with source URLs, retrieval times, response hashes, original date/stock-code fields, units, and report/remark meanings. Then verify a dated sample at the beginning and end of each year, all 100 training-stock identifiers and 50 target identifiers, duplicate keys, missing-row versus zero treatment, exchange transfers or eligibility changes, and whether later revisions can change the `t-2` value visible by prediction morning. Only after those checks should E9 feature construction and model comparison begin.
