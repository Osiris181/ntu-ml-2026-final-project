# Exploratory E11 Phase 1 submission workflow

E11's [historical validation report](E11_report.md) did **not** show a consistent gain over E3. This workflow creates an **exploratory** E11 submission at the user's request; it does not change the E3 reference decision or the E11 validation results.

Run these from the `final_project` directory, in order. Both scripts have the default paths already set:

```text
uv run python -m members.zhenci.e11.save_e11
uv run python -m members.zhenci.e11.predict_e11
```

[`save_e11.py`](save_e11.py) reuses the existing frozen E1 up-probability model and fitted F0 preprocessor, then trains E11's flat, conditional-gain, and conditional-loss XGBoost components on the 100-stock **2020-2025** historical training window. It freezes all four models, preprocessing, feature definitions, training caps, and source hashes in `results/zhenci_e11_expected_return_pre_phase1_artifact/`. It does not train on the January-May 2026 holdout or Phase 1 labels. It refuses to overwrite an existing artifact directory.

[`predict_e11.py`](predict_e11.py) loads only that frozen artifact, `data/processed/f0_with_updates.csv`, and `sample_submission.csv`. It applies the saved preprocessing and the unchanged E11 expected-return formula, checks every template stock/date key, and writes `submissions/zhenci_e11_expected_return_phase1.csv` in the official 50-stock-by-date layout. It does not refit any model or upload the file, and it refuses to overwrite an existing submission. The default template contains **4,200** stock/date predictions; a read-only preflight confirmed the updated F0 file currently contains all corresponding feature rows. The underlying F0 publication-timing and supplied-indicator assumptions remain as documented in E10.

If a default output already exists, pass a new `--output-dir` to `save_e11.py` and the matching `--artifact-dir` to `predict_e11.py`, or pass a new `--output` for another submission CSV. Do not replace the existing E1/E3 artifacts or prior submissions.

These scripts have been syntax-checked and their inputs preflighted, but **training and submission generation have not been run**. The final CSV should be inspected before any external upload.
