# 倪永珅

Personal experiment workspace. The primary approach remains unassigned.

Use this directory for your model scripts, configuration notes, and E1/E2 experiment explanations. For E1, load the shared [`F0 dataset`](../../data/processed/README.md), use the common [folds](../../shared/splits.py), fit [baseline preprocessing](../../shared/preprocessing.py) on each training fold, and score complete target predictions with [shared evaluation](../../shared/evaluation.py). Do not recreate the historical lag rules in a member script. See the [team guide](../../context/README.md) for inputs and outputs.
