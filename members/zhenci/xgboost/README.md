# XGBoost experiments

| Experiments | Files and reports |
| --- | --- |
| Baseline and early variants | [E1](e1/README.md), [E2](e2/README.md), [E3](e3/README.md), [E4](e4/README.md), [E5](e5/README.md), [E6](e6/README.md), [E7](e7/README.md) |
| Overrides, data feasibility, and expected return | [E8](e8/E8_report.md), [E9 feasibility](e9_data_feasibility.md), [E10](e10/), [E11](e11/E11_report.md) |
| XGBoost and LSTM ranking comparison | [E14](e14/E14_report.md) |
| Feature, target, and weighting studies | [E15](e15/E15_report.md), [E16](e16/E16_report.md), [E17](e17/E17_report.md), [E18](e18/E18_report.md) |
| Complexity and temporal adaptation | [E19](e19/E3_complexity_audit.md), [E20](e20/E20_report.md), [E21](e21/E21_report.md) |

See the [diagnostics](diagnostics/) for the E1/E3 and previous-direction comparisons and the [final report](conclusion/XGBOOST_FINAL_REPORT.md) for the research synthesis. E3’s [file guide](e3/README.md) includes its frozen Phase 2 exporter. E14 lives here because E3 is one of its two controls. The [LSTM workspace](../lstm/README.md) owns the sequence models.

Run scripts from the repository root using the commands in each experiment’s guide. For example, inspect the E3 Phase 2 exporter’s inputs with:

```powershell
uv run python -m members.zhenci.xgboost.e3.export_phase2 --help
```
