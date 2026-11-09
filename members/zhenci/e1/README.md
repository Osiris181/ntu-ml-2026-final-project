# E1: XGBoost baseline

This folder contains the controlled XGBoost baseline and its provisional Phase 1 artifact tools. It uses shared F0, chronological folds, five required validation seeds, fold-fitted baseline preprocessing, and the official scorer. See the [member overview](../README.md) for the full experiment description.

| File | Use |
|---|---|
| [`train_e1.py`](train_e1.py) | Run five seeds on the three validation folds and write metrics, probabilities, and a summary. |
| [`save_e1.py`](save_e1.py) | Refit a chosen E1 setting on pre-holdout training rows and freeze a provisional model and preprocessor. |
| [`predict_e1.py`](predict_e1.py) | Load that artifact and updated F0, check all official template keys, and export a Phase 1 CSV. |

Run from the repository root:

```text
uv run python -m members.zhenci.e1.train_e1
uv run python -m members.zhenci.e1.save_e1
uv run python -m members.zhenci.e1.predict_e1
```

The existing generated results and artifact paths under `results/` have not moved. `predict_e1.py` requires complete updated F0 inputs; it does not fetch data or upload a submission.
