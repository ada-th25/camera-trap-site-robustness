"""Plot the project's headline result: mAP50 dropping from seen to unseen
camera location, baseline vs fine-tuned.

Usage: PYTHONPATH=. .venv/bin/python scripts/plot_headline.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

OUT_DIR = Path("results/figures")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # From results/step2_results.md - already-verified numbers, not recomputed here.
    categories = ["seen location\n(val, loc 90)", "unseen location\n(test, loc 125)"]
    map50_values = [0.259, 0.122]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(categories, map50_values, color=[BLUE, ORANGE], width=0.5)
    for bar, val in zip(bars, map50_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color=TEXT_PRIMARY,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 0.32)
    ax.set_ylabel("mAP50", color=TEXT_PRIMARY)
    ax.set_title(
        "Detection accuracy drops by more than half\nat an unseen camera location",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "headline_seen_vs_unseen.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved to {OUT_DIR / 'headline_seen_vs_unseen.png'}")


if __name__ == "__main__":
    main()
