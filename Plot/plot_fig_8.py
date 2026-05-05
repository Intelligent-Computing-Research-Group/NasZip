
from __future__ import annotations

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

V_VARIANCE_DIR = os.path.join("../result", "Varience")
F_FREQ_DIR = os.path.join("../result", "early_stop_freq")

DATASETS: list[tuple[str, str]] = [
    ("SIFT", "SIFT.npy"),
    ("Wiki", "Wiki.npy"),
    ("GIST", "GIST.npy"),
    ("GloVe", "GloVe.npy"),
]


def load_v_f_pair(variance_fname: str, freq_fname: str) -> tuple[list, list]:
    v_path = os.path.join(V_VARIANCE_DIR, variance_fname)
    f_path = os.path.join(F_FREQ_DIR, freq_fname)
    v = np.load(v_path).tolist()
    f = np.load(f_path).tolist()
    return v, f


def plot_variance_vs_freq(
    out_path: str,
    datasets: list[tuple[str, str]] | None = None,
) -> None:
    datasets = datasets or DATASETS
    v_data: list[list] = []
    f_data: list[list] = []
    for _label, fname in datasets:
        v, f = load_v_f_pair(fname, fname)
        v_data.append(v)
        f_data.append(f)

    interval_list = [4, 16, 24, 5]
    labels = [lbl for lbl, _ in datasets]
    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    ax_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    show_yl_label = [True, False, True, False]
    show_yr_label = [False, True, False, True]

    for v, f, label, ax, interval, show_yl, show_yr in zip(
        v_data, f_data, labels, ax_list, interval_list, show_yl_label, show_yr_label
    ):
        x = np.arange(1, len(v) + 1)
        ax.plot(x, v, label="Variance", color="purple", lw=2)
        ax.set_xlabel(label, fontsize=18)
        if show_yl:
            ax.set_ylabel("Variance of\nRRE", fontsize=17)

        ax1b = ax.twinx()
        f = f[:-1][::interval]
        x = np.arange(1, 1 + len(f) * interval, interval)
        sum_f = sum(f)
        rat_f = 0.0
        ee_pos = 0
        for i in range(len(f)):
            f[i] = f[i] / sum_f
            rat_f += f[i]
            f[i] = rat_f
            if rat_f > 0.8 and ee_pos == 0:
                ee_pos = interval * i + 1

        f = [item * 100 for item in f]
        ax1b.plot(
            x,
            f,
            lw=2,
            label="Accumulated\nFEE Frequency",
            color="green",
            marker="v",
            markersize=4,
        )
        ax1b.axvline(x=ee_pos, linestyle="--", linewidth=3)
        if label != "GloVe":
            ax1b.text(x=ee_pos + 0.5, y=0.10 * max(f), s=f"80% FEE\nbefore\ndim{ee_pos}", fontsize=15)
        else:
            ax1b.text(x=ee_pos - 25, y=0.05 * max(f), s=f"80% FEE\nbefore\ndim{ee_pos}", fontsize=15)
        if show_yr:
            ax1b.set_ylabel("Accumulated\nFEE Frequency\n(%)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax1b.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    l1 = mlines.Line2D([], [], color="purple", label="Variance")
    l2 = mlines.Line2D([], [], color="green", marker="v", label="Accumulated FEE Frequency")
    fig.legend(
        handles=[l1, l2], loc="upper center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, 1.1), frameon=False
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_path = os.path.join("../Figures", "fig_8.pdf")
    plot_variance_vs_freq(out_path)


if __name__ == "__main__":
    main()
