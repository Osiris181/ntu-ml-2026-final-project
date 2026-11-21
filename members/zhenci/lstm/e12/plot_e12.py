"""Create E12 diagnostic SVGs from saved tables; never trains a model."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

DEFAULT = Path(__file__).resolve().parent / "outputs"


def learning_curves(frame: pd.DataFrame, path: Path) -> None:
    width,height=960,330
    colors={"train_bce":"#2866a3","outer_validation_bce_diagnostic_only":"#c15135"}
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="25" y="25" font-family="Arial" font-size="18">E12 fixed six-epoch BCE curves</text>']
    for i,year in enumerate(("2023","2024","2025")):
        group=frame.loc[frame["fold"].astype(str).eq(year)].sort_values("epoch")
        left=45+i*310; top=65; plot_w=250; plot_h=205
        xp=lambda epoch:left+(epoch-1)/5*plot_w
        yp=lambda loss:top+(.70-loss)/.03*plot_h
        lines.append(f'<text x="{left}" y="49" font-family="Arial" font-size="15">{year}</text>')
        for value in (.67,.68,.69,.70):
            y=yp(value)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            lines.append(f'<text x="{left-38}" y="{y+4:.1f}" font-family="Arial" font-size="10">{value:.2f}</text>')
        for epoch in range(1,7):
            x=xp(epoch)
            lines.append(f'<text x="{x-3:.1f}" y="288" font-family="Arial" font-size="10">{epoch}</text>')
        for field,color in colors.items():
            points=" ".join(f'{xp(row.epoch):.1f},{yp(row[field]):.1f}' for _,row in group.iterrows())
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    lines += ['<line x1="285" y1="314" x2="310" y2="314" stroke="#2866a3" stroke-width="3"/>',
              '<text x="316" y="318" font-family="Arial" font-size="12">Training BCE</text>',
              '<line x1="480" y1="314" x2="505" y2="314" stroke="#c15135" stroke-width="3"/>',
              '<text x="511" y="318" font-family="Arial" font-size="12">Outer validation BCE (diagnostic only)</text>',
              '</svg>']
    path.write_text("\n".join(lines),encoding="utf-8")


def prediction_balance(metrics: pd.DataFrame, probabilities: pd.DataFrame, path: Path) -> None:
    width,height=750,420
    left,top,plot_w,plot_h=65,55,635,285
    colors={"e1":"#777777","e3":"#c15135","e12":"#2866a3","actual":"#38814a"}
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="25" y="25" font-family="Arial" font-size="18">Predicted-up versus actual-up fractions</text>']
    for value in (0,.2,.4,.6):
        y=top+plot_h*(1-value/.6)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="18" y="{y+4:.1f}" font-family="Arial" font-size="11">{value:.0%}</text>')
    for i,year in enumerate(("2023","2024","2025")):
        base=left+95+i*205
        row=probabilities.loc[probabilities["fold"].astype(str).eq(year)].iloc[0]
        for j,arm in enumerate(("e1","e3","e12","actual")):
            value=(row["actual_up_rate"] if arm=="actual" else metrics.loc[
                metrics["fold"].astype(str).eq(year)&metrics["arm"].eq(arm),"predicted_up_rate"].iloc[0])
            bar_h=plot_h*value/.6
            x=base+(j-1.5)*30
            lines.append(f'<rect x="{x:.1f}" y="{top+plot_h-bar_h:.1f}" width="22" height="{bar_h:.1f}" fill="{colors[arm]}"/>')
        lines.append(f'<text x="{base-20}" y="365" font-family="Arial" font-size="13">{year}</text>')
    for i,arm in enumerate(("e1","e3","e12","actual")):
        x=140+i*145
        lines.append(f'<rect x="{x}" y="393" width="11" height="11" fill="{colors[arm]}"/>')
        lines.append(f'<text x="{x+17}" y="403" font-family="Arial" font-size="11">{escape(arm.upper() if arm!="actual" else "Actual")}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines),encoding="utf-8")


def run(output_dir: Path) -> None:
    learning_curves(pd.read_csv(output_dir/"learning_curves.csv"),output_dir/"learning_curves.svg")
    prediction_balance(pd.read_csv(output_dir/"fold_metrics.csv"),
                       pd.read_csv(output_dir/"probability_summary.csv"),
                       output_dir/"prediction_balance.svg")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT)
    args=parser.parse_args()
    run(args.output_dir)
