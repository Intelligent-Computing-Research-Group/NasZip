from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset_info = {
    "SIFT": {"size": "1M", "type": "L2", "dim": 128},
    "GIST": {"size": "1M", "type": "L2", "dim": 960},
    "BigANN": {"size": "1B", "type": "L2", "dim": 128},
    "Wiki": {"size": "1M", "type": "L2", "dim": 768},
    "GloVe": {"size": "1.2M", "type": "IP", "dim": 100},
    "MS_MARCO": {"size": "8M", "type": "L2", "dim": 384},
}


def read_and_process_data(csv_path: str, baseline_name: str, k: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"QPS": float})
    df = df[df["k"] == k]

    select_dataset = list(dataset_info.keys())
    df = df[df["dataset_name"].isin(select_dataset)]

    baseline_df = df[df["design_name"] == baseline_name]
    baseline_QPS = dict(zip(baseline_df["dataset_name"], baseline_df["QPS"]))
    df["QPS_norm"] = df.apply(lambda row: row["QPS"] / baseline_QPS[row["dataset_name"]], axis=1)

    QPS_df = df[["design_name", "dataset_name", "QPS_norm", "k"]].copy()
    QPS_df["Config"] = "Default"

    def _geomean_positive(s: pd.Series) -> float:
        valid = s[s > 0]
        if valid.empty:
            return float("nan")
        return float(np.exp(np.mean(np.log(valid))))

    geo = QPS_df.groupby("design_name")["QPS_norm"].apply(_geomean_positive).reset_index()
    geo["dataset_name"] = "GeoMean"
    geo["k"] = k
    geo["Config"] = "Default"

    QPS_df = pd.concat([QPS_df, geo], ignore_index=True)
    return QPS_df


def plot_scale(
    QPS_df_k_1: pd.DataFrame,
    QPS_df_k_10: pd.DataFrame,
    systems_order: list,
    dataset_order: list,
    custom_label: list,
) -> None:
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=BIGGER_SIZE)
    plt.rc("ytick", labelsize=BIGGER_SIZE)
    plt.rc("legend", fontsize=MEDIUM_SIZE)

    fig, ax1 = plt.subplots(figsize=(10, 3))
    plt.subplots_adjust(wspace=0.15)

    available_systems = [s for s in systems_order if s in QPS_df_k_1["design_name"].unique()]
    QPS_df_k_1_filtered = QPS_df_k_1[QPS_df_k_1["design_name"].isin(available_systems)].copy()
    QPS_df_k_10_filtered = QPS_df_k_10[QPS_df_k_10["design_name"].isin(available_systems)].copy()

    QPS_df_k_1_filtered = QPS_df_k_1[QPS_df_k_1["dataset_name"].isin(dataset_order)].copy()
    QPS_df_k_10_filtered = QPS_df_k_10[QPS_df_k_10["dataset_name"].isin(dataset_order)].copy()

    combined_df = pd.concat([QPS_df_k_1_filtered, QPS_df_k_10_filtered])

    combined_df["dataset_name_draw"] = combined_df.apply(
        lambda row: f"{row['dataset_name']}_{int(row['k'])}",
        axis=1,
    )

    colors = [
        "#D8BFD8",
        "#DEBB9D",
        "#F9E79F",
        "#ABEBC6",
        "#AED6F1",
        "#F4D03F",
        "#7DCEA0",
        "#85C1E9",
        "#F5B7B1",
    ]
    color_palette = dict(zip(available_systems, colors[: len(available_systems)]))

    bar_width = 0.65

    sns.barplot(
        x="dataset_name_draw",
        y="QPS_norm",
        hue="design_name",
        data=combined_df,
        ax=ax1,
        hue_order=available_systems,
        palette=color_palette,
        errorbar=None,
        edgecolor="black",
        linewidth=0.75,
        alpha=0.85,
        width=bar_width,
        dodge=True,
    )

    ax1.set_xlabel(None)
    xtick_labels = dataset_order + dataset_order
    ax1.set_xticks(range(len(xtick_labels)))
    ax1.set_xticklabels(xtick_labels, fontsize=20, rotation=90, ha="center")
    ax1.set_ylabel("Norm. Throughput", fontsize=20)
    ax1.tick_params(axis="x", pad=-0.5)
    plt.legend(
        labels=custom_label,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        fontsize=20,
        frameon=False,
    )

    plt.plot(
        [6.5, 6.5],
        [-1.5, 3.5],
        color="black",
        linestyle="--",
        linewidth=1,
        clip_on=False,
    )

    ax1.set_ylim(bottom=0.5, top=3.5)
    ax1.set_xlim(-0.5, len(dataset_order) * 2 - 0.5)

    ax1.text(2.5, -1.5, "recall@1", fontsize=24, ha="center", va="center", color="black")
    ax1.text(9.5, -1.5, "recall@10", fontsize=24, ha="center", va="center", color="black")

    data = combined_df.values.tolist()

    ax1.text(1 + bar_width * 0.5, ax1.get_ylim()[1] * 0.72, f"{data[13][2]:.2f}$\\times$", fontsize=20, rotation=90)
    ax1.text(3 + bar_width * 0.5, ax1.get_ylim()[1] * 0.72, f"{data[15][2]:.2f}$\\times$", fontsize=20, rotation=90)
    ax1.text(8 - bar_width * 0.67, ax1.get_ylim()[1] * 0.72, f"{data[28][2]:.2f}$\\times$", fontsize=20, rotation=90)
    ax1.text(8 + bar_width * 0.5, ax1.get_ylim()[1] * 0.72, f"{data[34][2]:.2f}$\\times$", fontsize=20, rotation=90)
    ax1.text(10 + bar_width * 0.5, ax1.get_ylim()[1] * 0.72, f"{data[36][2]:.2f}$\\times$", fontsize=20, rotation=90)

    os.makedirs("../Figures", exist_ok=True)
    plt.savefig("../Figures/fig_16.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["figure.titlesize"] = 20

    csv_path = os.path.join("../result", "Data", "overall_scaling.csv")
    QPS_df_k_1 = read_and_process_data(csv_path, "CPU-HP-Base", k=1)
    QPS_df_k_10 = read_and_process_data(csv_path, "CPU-HP-Base", k=10)
    plot_scale(
        QPS_df_k_1,
        QPS_df_k_10,
        ["CPU-HP-Base", "GPU", "MY-Scale"],
        ["SIFT", "GIST", "BigANN", "Wiki", "GloVe", "MS_MARCO", "GeoMean"],
        custom_label=["CPU-HP", "GPU", r"$\mathbf{NasZip(Scaled)}$"],
    )


if __name__ == "__main__":
    main()
