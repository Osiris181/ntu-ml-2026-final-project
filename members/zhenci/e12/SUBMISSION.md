# Exploratory E12 Phase 1 submission

E12's [validation report](E12_report.md) found mixed results and retained E3 as the reference. This workflow makes an **exploratory** E12 submission at the member's request. It does not change the E12 validation results or select a new LSTM configuration from Phase 1 feedback.

Run these commands from `final_project`, in order. Both scripts have default paths, so no flags are needed:

```powershell
uv run --extra e12 python -m members.zhenci.e12.save_e12
uv run --extra e12 python -m members.zhenci.e12.predict_e12
```

`save_e12.py` refits the same one-layer, 32-unit, six-epoch LSTM with seed 42 on all eligible 2020–2025 stock history. It uses the original shared `data/processed/f0.csv`, 20-session F0 sequences, and training-fitted median imputation, scaling, and categorical encodings. It saves the model, preprocessing state, feature and sequence definitions, threshold 0.5, and source hashes in `results/zhenci_e12_lstm_pre_phase1_artifact/`. It does not use 2026 labels, including the inspected holdout or Phase 1 period. It refuses to overwrite an existing artifact.

`predict_e12.py` reads that frozen artifact, `data/processed/f0_with_updates.csv`, and `sample_submission.csv`. For each template stock/date key, it builds a sequence ending at that date, applies the saved preprocessing and model, and writes `submissions/zhenci_e12_lstm_phase1.csv` in the official template layout. It uses no evaluation labels, does not refit, and refuses to overwrite an existing CSV. The resulting file is ready for **your review and upload**; the script does not upload it.

A read-only preflight found all 4,200 template stock/date rows and complete 20-session histories in the updated F0 file. The original and updated F0 files agree on pre-2026 feature values within numerical rounding tolerance. The final E12 refit and submission CSV have **not** been generated yet.

If a default artifact or CSV already exists, pass a new `--output-dir` to `save_e12.py` and the matching `--artifact-dir` to `predict_e12.py`, or use `--output` for a new submission path. The same F0 publication-timing and supplied technical-indicator assumptions documented for the shared pipeline still apply.
