# Configuration

`protocol.json` records the shared experiment settings: seed `42`, stock universes, the three expanding validation windows, the independent holdout, and the model-selection score. The corresponding split logic is in [`shared/splits.py`](../shared/splits.py).

All four E1 experiments use these settings. Keep algorithm-specific hyperparameters in the member's own workspace; no model configuration has been added here yet.
