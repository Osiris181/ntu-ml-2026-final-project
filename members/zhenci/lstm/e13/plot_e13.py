"""Render E13's saved learning curves; this script never trains a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT = Path(__file__).resolve().parent / "outputs"
COLORS = {5: "#2d8c82", 10: "#ce7226", 20: "#405fa6"}


def learning_curves(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.loc[frame["seed"].eq(42)].copy()
    if set(frame["window"]) != {5, 10, 20}:
        raise ValueError("Expected all three seed-42 windows")
    width, height = 1080, 415
    top, panel_height, panel_width = 65, 280, 285
    values = frame[["train_bce", "outer_validation_bce_diagnostic_only"]].to_numpy()
    low = float(values.min()) - .001
    high = float(values.max()) + .001
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="25" y="27" font-family="Arial" font-size="18">E13 seed 42: fixed six-epoch BCE curves</text>']
    for i, year in enumerate(("2023", "2024", "2025")):
        left = 65 + i * 340
        x_at = lambda epoch: left + (epoch - 1) / 5 * panel_width
        y_at = lambda value: top + (high - value) / (high - low) * panel_height
        lines.append(f'<text x="{left}" y="51" font-family="Arial" font-size="15">Validation {year}</text>')
        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = y_at(value)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="#e7e7e7"/>')
            lines.append(f'<text x="{left-46}" y="{y+4:.1f}" font-family="Arial" font-size="10">{value:.3f}</text>')
        for epoch in range(1, 7):
            lines.append(f'<text x="{x_at(epoch)-3:.1f}" y="363" font-family="Arial" font-size="10">{epoch}</text>')
        for window, color in COLORS.items():
            selected = frame.loc[frame["fold"].astype(str).eq(year) & frame["window"].eq(window)].sort_values("epoch")
            if len(selected) != 6:
                raise ValueError(f"Missing {window}-session curve for {year}")
            for field, dash in (("train_bce", ""), ("outer_validation_bce_diagnostic_only", ' stroke-dasharray="6 4"')):
                points = " ".join(f'{x_at(row.epoch):.1f},{y_at(row[field]):.1f}' for _, row in selected.iterrows())
                lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
    for index, (window, color) in enumerate(COLORS.items()):
        x = 155 + index * 140
        lines.append(f'<line x1="{x}" y1="395" x2="{x+22}" y2="395" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x+28}" y="399" font-family="Arial" font-size="11">{window} sessions</text>')
    lines += ['<line x1="605" y1="395" x2="627" y2="395" stroke="#444" stroke-width="2"/>',
              '<text x="635" y="399" font-family="Arial" font-size="11">Train</text>',
              '<line x1="701" y1="395" x2="723" y2="395" stroke="#444" stroke-width="2" stroke-dasharray="6 4"/>',
              '<text x="731" y="399" font-family="Arial" font-size="11">Outer validation (diagnostic)</text>',
              '</svg>']
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    learning_curves(pd.read_csv(args.output_dir / "learning_curves.csv"),
                    args.output_dir / "learning_curves.svg")
