"""Plot the fixed E20 and E3 boosting checkpoints from the saved metrics."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "outputs"


def plot(directory: Path) -> None:
    metrics = pd.read_csv(directory / "checkpoint_metrics.csv")
    means = metrics.groupby(["arm", "rounds"], as_index=False).mean(numeric_only=True)
    colors = {"E3": "#225c8e", "E20": "#b14b29"}
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.1), layout="constrained")
    for ax, field, label in zip(axes[0], ("bce", "auc", "accuracy"), ("Binary cross-entropy", "ROC AUC", "Direction accuracy")):
        ax.set_title(label, loc="left")
        for arm, group in means.groupby("arm"):
            for split, line in (("train", "-"), ("validation", "--")):
                ax.plot(group["rounds"], group[f"{split}_{field}"], line, color=colors[arm], marker="o", ms=3)
    for ax, field, label in zip(axes[1], ("validation_brier", "official_score", "mean_daily_spearman"),
                                ("Validation Brier", "Official score", "Mean daily Spearman")):
        ax.set_title(label, loc="left")
        for arm, group in means.groupby("arm"):
            ax.plot(group["rounds"], group[field], "--", color=colors[arm], marker="o", ms=3)
        ax.set_xlabel("Boosting rounds included")
    for ax in axes.flat:
        ax.grid(alpha=.23)
        ax.set_xticks([10, 25, 50, 75, 100])
    axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
    legend = [Line2D([0], [0], color=color, label=arm) for arm, color in colors.items()]
    legend += [Line2D([0], [0], color="#555", ls=line, label=split)
               for split, line in (("Training", "-"), ("Validation", "--"))]
    axes[0, 1].legend(handles=legend, loc="upper left", ncols=2, fontsize=8)
    fig.suptitle("E20 versus E3 across boosting checkpoints · equal-year means", fontsize=15)
    fig.savefig(directory / "boosting_curves.png", dpi=155, bbox_inches="tight")
    fig.savefig(directory / "boosting_curves.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    plot(parser.parse_args().output_dir)
