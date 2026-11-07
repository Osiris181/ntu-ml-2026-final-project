# Configuration

`protocol.json` records the shared experiment settings: the required E1 seeds `42`, `64`, `128`, `256`, and `512`, stock universes, the three expanding validation windows, the independent holdout, and the five-seed model-selection score. The `seed` field remains `42` as a reference for existing single-seed code. The corresponding split logic is in [`shared/splits.py`](../shared/splits.py); the protocol change is recorded in [`context/protocol-v1.2.md`](../context/protocol-v1.2.md).

All four E1 experiments use these settings. Keep algorithm-specific hyperparameters in the member's own workspace; no model configuration has been added here yet.
