"""Figure builders for Fig. 1 (leakage ablation) and Fig. 2 (per-class recall).

Both figures are sized for the single-column Springer LNCS text width.

Author: Amitabh Chakravorty
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"


def plot_ablation(n_removed, accuracy, f1_attack, path):
    """Grouped bars: accuracy and attack-class F1 after removing the k top-L* features."""
    x = np.asarray(n_removed)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width / 2, accuracy, width, color=BLUE, edgecolor="black",
           linewidth=0.6, label="Accuracy", zorder=3)
    ax.bar(x + width / 2, f1_attack, width, color=ORANGE, edgecolor="black",
           linewidth=0.6, label="Attack-class F1", zorder=3)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of features removed (descending $L^*$)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=15, fontweight="bold")
    ax.set_title("Leakage ablation on DS2OS (XGBoost)", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="lower left", fontsize=13, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_per_class(table, path):
    """table: DataFrame with columns label, recall, n_test, n_train, group.

    group is one of 'novel', 'rare' (fewer than 100 training rows) or 'seen'.
    """
    order = {"novel": 0, "rare": 1, "seen": 2}
    t = table.assign(_g=table["group"].map(order)).sort_values(["_g", "recall", "label"])
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, (_, r) in enumerate(t.iterrows()):
        color = RED if r.group == "novel" else BLUE
        hatch = "//" if r.group == "rare" else None
        ax.bar(i, r.recall, color=color, edgecolor="black", linewidth=0.6, hatch=hatch, zorder=3)
        note = f"n={int(r.n_test)}"
        if r.group == "rare":
            note += f"\ntrain={int(r.n_train)}"
        ax.text(i, r.recall + 0.015, note, ha="center", va="bottom", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(t["label"], rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Recall on KDDTest+", fontsize=14)
    ax.set_title("Per-class Recall on the NSL-KDD Holdout (XGBoost)", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(handles=[
        Patch(facecolor=RED, edgecolor="black", label="Novel (absent from KDDTrain+)"),
        Patch(facecolor=BLUE, edgecolor="black", label="Seen in training"),
        Patch(facecolor=BLUE, edgecolor="black", hatch="//", label="Seen, under 100 training instances"),
    ], loc="upper left", ncol=3, fontsize=10.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
