# Data

| Path | Use |
|---|---|
| [`info.json`](info.json) | Lists the 100 supplied stocks (`Top100@Y`) and the 50 evaluation targets (`Top50@Y`). |
| [`market/`](market/README.md) | Supplied stock history, one CSV per stock. Includes historical labels for training and scoring. |
| [`global/`](global/README.md) | Supplied global market history. |
| [`processed/`](processed/README.md) | Rebuildable, shared F0 feature dataset. |
| [`raw/`](raw/README.md) | Placeholder for separately supplied raw inputs; the current stock and global CSVs live in the folders above. |
| [`splits/`](splits/README.md) | Explains the date split definitions. |

The source CSVs and `info.json` are inputs and should stay unchanged. [`shared/data_loader.py`](../shared/data_loader.py) validates and loads them; [`shared/features.py`](../shared/features.py) builds the common feature rows. The central timing and indicator assumptions are documented in the [project README](../README.md#data-quality-and-timing-assumptions).
