# LSTM experiments

- [E12 report](e12/E12_report.md) covers the 20-session LSTM baseline. Its [inference guide](e12/SUBMISSION.md) documents saving and loading the frozen model.
- [E13 report](e13/E13_report.md) covers the 5-, 10-, and 20-session comparison and its multi-seed check.
- [E14 report](../xgboost/e14/E14_report.md) compares daily ranking for E12 and E3 using their saved probabilities.

Run from the repository root with the `e12` dependency extra. For example, inspect the E12 inference options with:

```powershell
uv run --extra e12 python -m members.zhenci.lstm.e12.predict_e12 --help
```
