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

Complete E1 first. Select its configuration by the shared CV score, save a refitted E1 model and fitted preprocessor, and prepare a submission-ready Phase 1 CSV once validated inputs cover the official template. Coordinate the actual upload under the competition's submission rules. Start E2 afterward and compare it with your E1 on the shared validation folds; do not use submission feedback to tune E2. The [submission guide](../submissions/README.md) explains the file format.
