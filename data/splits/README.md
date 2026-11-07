# Split definitions

This directory currently contains this explanation; there are no separate split CSVs. The versioned date windows and seed `42` are in [`configs/protocol.json`](../../configs/protocol.json). [`shared/splits.py`](../../shared/splits.py) implements the three expanding training/validation folds and the Jan–May 2026 independent holdout.

The split functions intersect each window with actual supplied trading rows. Training includes all 100 stocks; evaluation keys are the target 50 stock/date rows present in the source data. Use those same keys for every approach.

Select configurations using the equally weighted mean of the three yearly validation scores. Do not use the Jan–May 2026 holdout for configuration selection.
