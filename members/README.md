# Member workspaces

Each subdirectory uses the romanized given name:

| Member | Directory |
|---|---|
| 倪永珅 | [`yongshen/`](yongshen/README.md) |
| 葉家堯 | [`jiayao/`](jiayao/README.md) |
| 鄭振慈 | [`zhenci/`](zhenci/README.md) |
| 龔建瑄 | [`jianxuan/`](jianxuan/README.md) |

鄭振慈 has chosen XGBoost; the other primary approaches are still unassigned.

For E1, every member uses the shared [`F0 dataset`](../data/processed/README.md), [`chronological folds and five seeds`](../shared/splits.py), [`BaselinePreprocessor`](../shared/preprocessing.py), and [`official scorer and five-seed summary`](../shared/evaluation.py). The required seeds are `42`, `64`, `128`, `256`, and `512`. Run all three folds for each seed and retain all 15 scores. Keep model-specific code and E2 changes in your own directory. The [team guide](../context/README.md) shows the shared inputs, outputs, and workflow.
