# Member workspaces

The four subdirectories belong to 倪永珅, 葉家堯, 鄭振慈, and 龔建瑄. 鄭振慈 has chosen XGBoost and has a prepared E1 runner; the other primary approaches are still unassigned. No model has been trained and no experiment results have been generated yet.

For E1, every member uses the shared [`F0 dataset`](../data/processed/README.md), [`chronological folds`](../shared/splits.py), [`BaselinePreprocessor`](../shared/preprocessing.py), seed `42`, and [`official scorer`](../shared/evaluation.py). Keep model-specific code and E2 changes in your own directory. The [team guide](../context/README.md) shows the shared inputs, outputs, and workflow.
