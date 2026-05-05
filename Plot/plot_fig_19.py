from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CSV 中的 design_name，顺序与 notebook 中 sift_uniq_designs 一致
DESIGN_ORDER = ["CPU", "SCANN", "UPMEM", "PIMANN", "ANSMET", "NasZip"]
# 图例显示名（对应 notebook 中对 HNSW / UPMEM / NasZip 的展示习惯）
LEGEND_LABELS = ["HNSW", "SCANN", "UPMEM+FEE-sPCA", "PIMANN", "ANSMET", "NasZip"]


def _parse_qps(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.strip()
        .astype(float)
    )


def load_qps_recall_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["QPS"] = _parse_qps(df["QPS"])
    return df


def build_recall_vs_kqps(
    df: pd.DataFrame,
    dataset_name: str,
    design_order: list[str],
    recall_min: float | None = None,
) -> list[tuple[str, list[float], list[float]]]:
    sub = df[df["dataset_name"] == dataset_name]
    out: list[tuple[str, list[float], list[float]]] = []
    for design in design_order:
        design_records = sub[sub["design_name"] == design]
        if design_records.empty:
            continue
        xy = design_records[["recall", "QPS"]].copy()
        if recall_min is not None:
            xy = xy[xy["recall"] >= recall_min]
        if xy.empty:
            continue
        xy = xy.sort_values(["recall", "QPS"], kind="mergesort")
        recall_values = xy["recall"].tolist()
        qps_values = (xy["QPS"] / 1000.0).tolist()
        out.append((design, recall_values, qps_values))
    return out


def plot_qps_recall(
    csv_path: str,
    out_path: str,
    design_order: list[str] | None = None,
    legend_labels: list[str] | None = None,
) -> None:
    design_order = design_order or DESIGN_ORDER
    legend_labels = legend_labels or LEGEND_LABELS
    if len(legend_labels) != len(design_order):
        raise ValueError("legend_labels 必须与 design_order 等长")

    df = load_qps_recall_csv(csv_path)

    sift_curves = build_recall_vs_kqps(df, "SIFT", design_order)
    glove_curves = build_recall_vs_kqps(df, "GloVe", design_order, recall_min=0.8)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    colors = [
        "#589dda",
        "#ea7f34",
        "#6aa751",
        "#ff595e",
        "#9467bd",
        "#8c564b",
    ]

    for line_data in sift_curves:
        design, recall_values, qps_values = line_data
        idx = design_order.index(design)
        axs[0].plot(
            recall_values,
            qps_values,
            label=design,
            marker="v",
            linewidth=3,
            color=colors[idx % len(colors)],
        )
    axs[0].set_xlabel("Recall@10(SIFT)", fontsize=24)
    axs[0].set_ylabel("KQPS", fontsize=24)
    axs[0].legend(fontsize=18)
    axs[0].tick_params(axis="both", labelsize=22)

    for line_data in glove_curves:
        design, recall_values, qps_values = line_data
        idx = design_order.index(design)
        axs[1].plot(
            recall_values,
            qps_values,
            label=design,
            marker="v",
            linewidth=3,
            color=colors[idx % len(colors)],
        )
    axs[1].set_xlabel("Recall@10(GloVe)", fontsize=24)
    # axs[1].set_xlim(0.81, 1.02)
    axs[1].legend(fontsize=18)
    axs[1].tick_params(axis="both", labelsize=24)

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axs[0].yaxis.offsetText.set_fontsize(19)
    axs[1].yaxis.offsetText.set_fontsize(19)

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    # 与 notebook 一致：用自定义图例名覆盖
    labels = list(labels)
    name_to_display = dict(zip(design_order, legend_labels))
    labels = [name_to_display.get(lab, lab) for lab in labels]

    axs[0].grid(visible=True, which="both", axis="both", color="gray", linestyle="--", linewidth=2, alpha=0.3)
    axs[1].grid(visible=True, which="both", axis="both", color="gray", linestyle="--", linewidth=1, alpha=0.4)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        handlelength=1,
        handletextpad=0.2,
        ncol=3,
        fontsize=20,
        bbox_to_anchor=(0.5, 1.2),
        columnspacing=2,
        frameon=False,
    )
    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout(pad=2.0, w_pad=0.1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csv_path = os.path.join("../result", "Data", "qps_vs_recall.csv")
    plot_qps_recall(csv_path, out_path=os.path.join("../Figures", "fig_19.pdf"))


if __name__ == "__main__":
    main()
