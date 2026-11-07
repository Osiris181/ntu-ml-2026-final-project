# 倪永珅

Personal experiment workspace. The primary approach remains unassigned.

Use this directory for your model scripts, configuration notes, and E1/E2 experiment explanations. For E1, load the shared [`F0 dataset`](../../data/processed/README.md), use the common [folds](../../shared/splits.py), fit [baseline preprocessing](../../shared/preprocessing.py) on each training fold, and score complete target predictions with [shared evaluation](../../shared/evaluation.py). Do not recreate the historical lag rules in a member script. See the [team guide](../../context/README.md) for inputs and outputs.

Run E1 first, select its configuration using the shared five-seed CV score, and prepare a submission-ready Phase 1 baseline file from a refitted E1 artifact when validated inputs cover every official template key. Coordinate any upload under the competition's rules. Then develop E2 and compare it with your E1 on the shared folds; do not tune E2 from submission feedback.
