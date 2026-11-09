"""Export a provisional E3 regime artifact using the official Phase 1 template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from members.zhenci.e1.predict_e1 import export_predictions
from members.zhenci.e3.save_e3_regime import DEFAULT_CV_SUMMARY, DEFAULT_OUTPUT_DIR
from members.zhenci.e3.train_e3_regime import CANDIDATES


def require_artifact(artifact_dir: Path) -> None:
    """Explain why there is no E3 model before trying to read its files."""
    if (artifact_dir / "manifest.json").is_file():
        return
    if artifact_dir == DEFAULT_OUTPUT_DIR and DEFAULT_CV_SUMMARY.is_file():
        with DEFAULT_CV_SUMMARY.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        if summary.get("selected_config_id") == CANDIDATES[0]["config_id"]:
            raise FileNotFoundError(
                "E3 regime selected the E1 control, so save_e3_regime did not create "
                "an alternative model. There is no E3 artifact to predict with."
            )
    raise FileNotFoundError(
        f"No saved E3 model at {artifact_dir}. Run save_e3_regime after "
        "a non-baseline E3 configuration wins validation."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0_with_updates.csv")
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "sample_submission.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submissions/zhenci_e3_regime_phase1.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    require_artifact(args.artifact_dir)
    print(export_predictions(args.artifact_dir, args.f0_path, args.template, args.output, args.data_dir))
