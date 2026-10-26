"""Plot recall-by-condition charts from scripts/analyze_robustness.py output.

Usage: PYTHONPATH=. .venv/bin/python scripts/plot_robustness.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE = "#2a78d6"

OUT_DIR = Path("results/figures")


def bar_chart(categories, values, counts, title, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(categories, values, color=BLUE, width=0.5)
    for bar, val, n in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            color=TEXT_PRIMARY,
            fontsize=9,
        )

    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("Recall", color=TEXT_PRIMARY)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    box_df = pd.read_csv("data/processed/robustness_breakdown.csv")

    by_daynight = box_df.groupby("day_night")["found"].agg(["mean", "count"])
    bar_chart(
        by_daynight.index.tolist(),
        by_daynight["mean"].tolist(),
        by_daynight["count"].tolist(),
        "Recall: day vs. night/IR (test set, unseen location)",
        "capture condition",
        OUT_DIR / "recall_by_daynight.png",
    )

    by_size = (
        box_df.groupby("size_bucket")["found"].agg(["mean", "count"]).reindex(["small", "medium", "large"])
    )
    bar_chart(
        by_size.index.tolist(),
        by_size["mean"].tolist(),
        by_size["count"].tolist(),
        "Recall by animal size (test set, unseen location)",
        "bbox size (fraction of image area)",
        OUT_DIR / "recall_by_size.png",
    )

    print(f"Saved charts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
