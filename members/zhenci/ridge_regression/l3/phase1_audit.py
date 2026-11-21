"""Verify existing Phase 1 files after the L3 historical choice has been recorded."""

from pathlib import Path

import numpy as np
import pandas as pd

from members.zhenci.logistic_regression.l1.export_phase1 import DATA, PHASE1, PHASE1_B, TEMPLATE
from members.zhenci.logistic_regression.l1.model import digest, read_json
from members.zhenci.logistic_regression.l2.export_phase1 import PHASE1 as L2_PHASE1
from shared.data_loader import KEYS, load_metadata, validate_keys
from shared.submission import load_submission_template

ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "members/zhenci/xgboost/diagnostics/e1_e3_gap_outputs"
CLAIMS = {
    "E1": (0.00058145, -0.00031791, 8, "zhenci_e1_phase1.csv"),
    "E3": (0.00064050, -0.00006194, 8, "zhenci_e3_taiex_phase1.csv"),
    "L1-A": (0.00047616, -0.000119, 6, "zhenci_l1_a_exploratory_phase1.csv"),
    "L1-B": (0.00048060, 0.000304, 6, "zhenci_l1_b_exploratory_phase1.csv"),
    "L2-A": (0.00054579, None, None, None),
    "L2-B": (0.00054920, 0.000659, 6, "zhenci_l2_b_exploratory_phase1.csv"),
}


def audit_prior_submissions(overall):
    targets, _ = load_metadata(DATA)
    ids, dates = load_submission_template(TEMPLATE, targets)
    keys = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS)
    path = REPLAY / "paired_rows.csv"
    previous = pd.read_csv(path, dtype={"stock_id": str}, parse_dates=["date"])
    validate_keys(previous)
    if len(previous) != len(keys) or not keys.isin(pd.MultiIndex.from_frame(previous[KEYS])).all():
        raise ValueError("Previous Phase 1 replay differs from the official template")
    previous = previous.set_index(KEYS).reindex(keys)
    returns = previous["realized_return"].to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("Previous Phase 1 replay lacks realized returns")
    previous_summary = read_json(REPLAY / "summary.json")
    for arm in ("e1", "e3"):
        np.testing.assert_allclose(((2 * previous[f"{arm}_pred"] - 1) * returns).mean(),
                                   previous_summary[arm]["official_score"], rtol=0, atol=1e-14)
    manifests = {"L1-A": PHASE1 / "export_manifest.json", "L1-B": PHASE1_B / "export_manifest.json",
                 "L2-B": L2_PHASE1 / "export_manifest.json"}
    official = overall.set_index("arm")["official_score"]
    sources = [path, REPLAY / "summary.json", TEMPLATE, *manifests.values()]
    rows = []
    for arm, (historical, submitted, digits, name) in CLAIMS.items():
        if round(official.at[arm], 8) != historical:
            raise ValueError(f"Archived {arm} historical score does not match the brief")
        row = dict(arm=arm, archived_cv_score=official.at[arm], reported_phase1_score=submitted,
                   locally_replayed_phase1_score=np.nan, rounded_match=None, submission=name)
        if name is not None:
            csv = ROOT / "submissions" / name
            sources.append(csv)
            if arm in manifests:
                meta = read_json(manifests[arm])
                expected = meta["submission_sha256"]
            else:
                expected = previous_summary[f"{arm.lower()}_submission_sha256"]
            if digest(csv) != expected:
                raise ValueError(f"Saved {arm} submission differs from its archive")
            predictions = pd.read_csv(csv, dtype={"id": str})
            if (predictions.columns.tolist() != ["id", *dates] or predictions["id"].tolist() != ids
                    or not predictions[dates].isin([0, 1]).all().all()):
                raise ValueError(f"Saved {arm} submission differs from the Phase 1 template")
            calls = predictions[dates].to_numpy().ravel()
            score = float(np.mean((2 * calls - 1) * returns))
            row.update(locally_replayed_phase1_score=score, rounded_match=round(score, digits) == submitted)
        rows.append(row)
    return pd.DataFrame(rows), {
        "purpose": "Verify prior claims after selection, using the archived adjusted-return replay",
        "competition_server_receipts": "unavailable",
        "input_sha256": {file.relative_to(ROOT).as_posix(): digest(file) for file in sources},
    }
