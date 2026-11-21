"""Render E19 diagnostic curves from saved CSVs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLORS = {2023: "#205787", 2024: "#ad4c24", 2025: "#287551"}
PRIMARY = [10, 25, 50, 75, 100]


def save(fig, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def curves(directory: Path) -> None:
    frame = pd.read_csv(directory / "checkpoint_metrics.csv")
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.0), layout="constrained")
    titles = ("Binary cross-entropy", "ROC AUC", "Direction accuracy", "Validation Brier score",
              "Validation official score", "Mean daily validation Spearman", "Validation predicted-up fraction",
              "Validation probability spread", "Fraction of validation probabilities outside [0.1, 0.9]")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(PRIMARY)
        ax.grid(alpha=.22)
    for year, color in COLORS.items():
        y = frame.loc[frame["fold"].eq(year)].sort_values("rounds")
        x = y["rounds"]
        for i, field in enumerate(("bce", "auc", "accuracy")):
            axes.flat[i].plot(x, y[f"train_{field}"], color=color, lw=1.7)
            axes.flat[i].plot(x, y[f"validation_{field}"], color=color, ls="--", lw=1.7)
        for i, field in enumerate(("validation_brier", "validation_official_score", "validation_mean_daily_spearman",
                                    "validation_predicted_up_rate"), start=3):
            axes.flat[i].plot(x, y[field], color=color, ls="--", lw=1.7)
        axes.flat[7].plot(x, y["validation_p50"], color=color, lw=1.7)
        axes.flat[7].fill_between(x, y["validation_p10"], y["validation_p90"], color=color, alpha=.13)
        axes.flat[8].plot(x, y["validation_p_lt_010"] + y["validation_p_gt_090"], color=color, ls="--", lw=1.7)
    axes.flat[4].ticklabel_format(style="sci", axis="y", scilimits=(-3,-3))
    axes.flat[7].text(.03,.95,"Median and 10th–90th percentile band",transform=axes.flat[7].transAxes,va="top",fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Boosting rounds included")
    legend = [Line2D([0],[0], color=color,label=str(year)) for year,color in COLORS.items()]
    legend.extend([Line2D([0],[0],color="#555",label="Training"),
                   Line2D([0],[0],color="#555",ls="--",label="Validation")])
    fig.suptitle("E19 diagnostic curves from frozen E3 models", fontsize=15)
    axes[0,1].legend(handles=legend,loc="upper left",ncol=2,fontsize=8)
    save(fig,directory,"boosting_curves")


def stability(directory: Path) -> None:
    months = pd.read_csv(directory / "monthly_scores.csv")
    blocks = pd.read_csv(directory / "four_month_scores.csv")
    limit = np.abs(months["official_score"]).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(3, 2, figsize=(11,9), layout="constrained", width_ratios=[2.3, 1])
    for row, year in enumerate(COLORS):
        for col, (source, period) in enumerate(((months,"month"),(blocks,"block"))):
            grid = source.loc[source["fold"].eq(year)].pivot(index="rounds",columns=period,values="official_score")
            grid = grid.reindex(PRIMARY)
            ax = axes[row,col]
            picture = ax.imshow(grid.to_numpy(),cmap="RdBu",norm=norm,aspect="auto")
            labels = [label.split("-")[-1] for label in grid.columns]
            ax.set_xticks(range(len(labels)), labels=labels)
            ax.set_yticks(range(len(PRIMARY)),labels=PRIMARY)
            ax.set_ylabel(f"{year} rounds")
            if col == 1:
                ax.set_xticks(range(3), labels=["Jan–Apr","May–Aug","Sep–Dec"])
    axes[0,0].set_title("Calendar months")
    axes[0,1].set_title("Fixed four-month blocks")
    fig.colorbar(picture,ax=axes.ravel().tolist(),label="Official score",shrink=.76)
    fig.suptitle("E19 score stability across validation periods",fontsize=15)
    save(fig,directory,"period_stability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=HERE/"outputs")
    directory = parser.parse_args().output_dir
    curves(directory)
    stability(directory)
