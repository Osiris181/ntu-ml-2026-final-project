# Stock market data

This directory contains one supplied `<stock_id>.csv` per stock, for 100 stocks. Each row has a trading `date`, adjusted prices, trading activity, institutional fields, supplied technical indicators, and the historical binary label `y`.

[`shared/data_loader.py`](../../shared/data_loader.py) loads these files using the IDs in [`../info.json`](../info.json) and checks their keys and labels. [`shared/features.py`](../../shared/features.py) uses previous completed stock sessions to build F0. The target day's `y`, prices, and other completed-session values are never F0 inputs. Training code obtains labels and evaluation prices separately from these source rows, joining by `(stock_id, date)`.

Keep the supplied CSVs unchanged. Their dates are observation dates, not verified publication timestamps; see the [central data-quality section](../../README.md#data-quality-and-timing-assumptions).
