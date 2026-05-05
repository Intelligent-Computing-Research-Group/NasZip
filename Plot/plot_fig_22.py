from __future__ import annotations

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# 相对仓库工作目录；与 notebook 中 Results/pf_hit_rate 对应
PREFETCH_HIT_FILES = {
    16: os.path.join("../result", "prefetch", "SIFT_16_pf_hit.npy"),
    32: os.path.join("../result", "prefetch", "SIFT_32_pf_hit.npy"),
    64: os.path.join("../result", "prefetch", "SIFT_64_pf_hit.npy"),
}

# cache_hit_rate.csv：cache_size -> 图例（与原 Excel 列名一致）
CACHE_SIZE_KB_LABELS: dict[int, str] = {
    500: "32KB",
    1000: "64KB",
    2000: "128KB",
    4000: "256KB",
}

CACHE_SIZES_ORDER = [500, 1000, 2000, 4000]


def load_prefetch_hits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit_16 = np.load(PREFETCH_HIT_FILES[16])
    hit_32 = np.load(PREFETCH_HIT_FILES[32])
    hit_64 = np.load(PREFETCH_HIT_FILES[64])
    return hit_16, hit_32, hit_64


def scale_prefetch_to_percent(
    hit_16: np.ndarray, hit_32: np.ndarray, hit_64: np.ndarray, n: int = 50
) -> None:
    n = min(n, len(hit_16), len(hit_32), len(hit_64))
    hit_16[:n] *= 100.0
    hit_32[:n] *= 100.0
    hit_64[:n] *= 100.0


def load_cache_hit_curves(
    csv_path: str, dataset_name: str = "SIFT"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    df = df[df["dataset_name"] == dataset_name]
    search_ef: np.ndarray | None = None
    curves: dict[str, np.ndarray] = {}
    for size in CACHE_SIZES_ORDER:
        sub = df[df["cache_size"] == size].sort_values("ef", kind="mergesort")
        if sub.empty:
            continue
        label = CACHE_SIZE_KB_LABELS[size]
        if search_ef is None:
            search_ef = sub["ef"].to_numpy()
        rates = sub["cache_hit_rate"].to_numpy(dtype=float) * 100.0
        curves[label] = rates
    if search_ef is None or not curves:
        raise ValueError(f"CSV 中无 dataset_name={dataset_name!r} 的 cache 曲线数据")
    return search_ef, curves


def plot_detail_4(
    cache_csv_path: str,
    out_path: str,
    dataset_name: str = "SIFT",
) -> None:
    hit_16, hit_32, hit_64 = load_prefetch_hits()
    scale_prefetch_to_percent(hit_16, hit_32, hit_64, n=50)

    search_ef, cache_curves = load_cache_hit_curves(cache_csv_path, dataset_name=dataset_name)

    colors = ["#C10505", "#0Dadba", "#0280cc", "#F8B62C"]

    fig, axs = plt.subplots(1, 2, figsize=(11, 3))
    plt.subplots_adjust(wspace=0.4)

    axs[1].plot(range(1, 50), hit_16[1:50], marker="v", markersize=3, color="blue")
    axs[1].plot(range(1, 44), hit_32[1:44], marker="v", markersize=3, color="green")
    axs[1].plot(range(1, 36), hit_64[1:36], marker="v", markersize=3, color="orange")

    h16 = mlines.Line2D([], [], color="blue", marker="v", label=r"$efCon.16$")
    h32 = mlines.Line2D([], [], color="green", marker="v", label=r"$efCon.32$")
    h64 = mlines.Line2D([], [], color="orange", marker="v", label=r"$efCon.64$")

    axs[1].set_xlabel("Hops", fontsize=20)
    axs[1].set_ylabel("Prefetch Hit Rate (%)", fontsize=18)
    axs[1].tick_params(axis="both", labelsize=18)
    axs[1].set_xlim(0, 50)
    axs[1].grid(which="major", linestyle="-", linewidth=0.5, alpha=0.7)
    axs[1].axvspan(0, 9.5, color="lightpink", alpha=0.4)
    axs[1].axvspan(9.5, 50, color="lightgreen", alpha=0.4)

    legend_ele = [
        Patch(facecolor="lightpink", alpha=0.4, edgecolor="none", label="upper layer"),
        Patch(facecolor="lightgreen", alpha=0.4, edgecolor="none", label="base layer"),
    ]
    leg1 = axs[1].legend(
        handles=[h16, h32, h64],
        ncol=1,
        fontsize=18,
        labelspacing=0.3,
        handletextpad=0.1,
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.05, 0.5),
    )
    axs[1].add_artist(leg1)
    axs[1].legend(
        handles=legend_ele,
        loc="upper center",
        labelspacing=0.4,
        columnspacing=0.7,
        markerscale=0.5,
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.4, 1.25),
        fontsize=18,
    )

    label_order = [CACHE_SIZE_KB_LABELS[s] for s in CACHE_SIZES_ORDER if CACHE_SIZE_KB_LABELS[s] in cache_curves]
    for label, color in zip(label_order, colors):
        y = cache_curves[label]
        axs[0].plot(
            search_ef,
            y,
            marker="D",
            label=label,
            linewidth=1.5,
            markersize=3,
            color=color,
        )

    axs[0].set_xlabel(r"$efSearch$", fontsize=20)
    axs[0].set_ylabel("Cache Hit Rate (%)", fontsize=18)
    axs[0].tick_params(axis="both", labelsize=18)
    axs[0].grid(which="major", linestyle="-", linewidth=0.5, alpha=0.7)
    axs[0].legend(
        ncol=4,
        loc="upper center",
        handlelength=1,
        handletextpad=0.3,
        bbox_to_anchor=(0.4, 1.25),
        frameon=False,
        fontsize=16,
        columnspacing=0.8,
    )

    fig.text(0.05, -0.1, "(a)", fontsize=20)
    fig.text(0.5, -0.1, "(b)", fontsize=20)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache_csv = os.path.join("../result", "cache_hit_rate.csv")
    out_path = os.path.join("../Figures", "fig_22.pdf")
    plot_detail_4(cache_csv, out_path)


if __name__ == "__main__":
    main()
