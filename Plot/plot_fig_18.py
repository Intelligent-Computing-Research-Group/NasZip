from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_ORDER = ["SIFT", "GIST", "BigANN", "Wiki", "GloVe", "MS_MARCO"]
DESIGN_ORDER = ["CPU-Base", "NDP-Base", "ANSMET", "MY"]
DESIGN_LABELS = ["CPU", "NDP-Base", "ANSMET", r"$\mathbf{NasZip}$"]


def calculate_geomean(df: pd.DataFrame, metric_col: str) -> float:
    valid_values = df[metric_col].replace(0, np.nan).dropna().values
    if len(valid_values) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid_values))))


def load_prepare(
    csv_path: str,
    baseline_name: str = "MY",
    dataset_order: list[str] | None = None,
    design_order: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "latency_10us": "latency",
            "dist_cal": "dist_cal_latency_ratio",
            "nbr_list_fetch": "nbr_fetch_latency_ratio",
            "parital_res_processing": "result_collection_latency_ratio",
        }
    )
    dorder = dataset_order or DATASET_ORDER
    df = df[df["dataset_name"].isin(dorder)].copy()

    all_designs = design_order or DESIGN_ORDER
    df = df[df["design_name"].isin(all_designs)].copy()

    geomean_rows = []
    for design in all_designs:
        design_data = df[df["design_name"] == design]
        if design_data.empty:
            continue
        geomean_latency = calculate_geomean(design_data, "latency")
        geomean_rows.append(
            {
                "design_name": design,
                "dataset_name": "GeoMean",
                "latency": geomean_latency,
                "dist_cal_latency_ratio": design_data["dist_cal_latency_ratio"].mean(),
                "nbr_fetch_latency_ratio": design_data["nbr_fetch_latency_ratio"].mean(),
                "result_collection_latency_ratio": design_data["result_collection_latency_ratio"].mean(),
            }
        )
    geomean_df = pd.DataFrame(geomean_rows)
    df = pd.concat([df, geomean_df], ignore_index=True)

    full_dataset_order = list(dorder) + ["GeoMean"]
    normalized_df = df.copy()
    for dataset in full_dataset_order:
        group_df = df[df["dataset_name"] == dataset]
        baseline_record = group_df[group_df["design_name"] == baseline_name]
        if baseline_record.empty or baseline_record["latency"].values[0] == 0:
            continue
        baseline_value = float(baseline_record["latency"].values[0])
        group_mask = df["dataset_name"] == dataset
        normalized_df.loc[group_mask, "normalized_latency"] = df.loc[group_mask, "latency"] / baseline_value
    return normalized_df


def plot_latency_breakdown(
    df: pd.DataFrame,
    dataset_order: list[str] | None = None,
    design_order: list[str] | None = None,
    design_labels: list[str] | None = None,
    out_path: str = "../Figures/fig_18.pdf",
) -> None:
    dorder = dataset_order or (DATASET_ORDER + ["GeoMean"])
    ds_order = design_order or DESIGN_ORDER
    dlabels = design_labels or DESIGN_LABELS

    latency_colors = {
        "dist_cal_latency_ratio": "#F4D03F",
        "nbr_fetch_latency_ratio": "#76B7B2",
        "result_collection_latency_ratio": "#AEC7E8",
    }

    fig, ax = plt.subplots(figsize=(12, 3))
    bar_width = 0.2
    begin_offset = 0.1

    for design_idx, design_name in enumerate(ds_order):
        group_data = df[df["design_name"] == design_name]
        for dataset_idx, dataset_name in enumerate(dorder):
            row = group_data.loc[group_data["dataset_name"] == dataset_name]
            if row.empty:
                continue
            latency = float(row["normalized_latency"].values[0])
            dist_cal_latency_ratio = float(row["dist_cal_latency_ratio"].values[0])
            nbr_fetch_latency_ratio = float(row["nbr_fetch_latency_ratio"].values[0])
            result_collection_latency_ratio = float(row["result_collection_latency_ratio"].values[0])

            cur_pos = begin_offset + dataset_idx + design_idx * bar_width

            if design_name == "CPU-Base" or (
                (dataset_name == "GIST" or dataset_name == "Wiki") and design_name == "NDP-Base"
            ):
                if design_name == "CPU-Base" and (dataset_name == "GIST" or dataset_name == "BigANN"):
                    ax.text(
                        cur_pos,
                        4.5,
                        f"{latency:.1f}$\\times$",
                        ha="center",
                        va="bottom",
                        fontsize=17,
                        color="black",
                        rotation=90,
                    )
                elif design_name == "CPU-Base" and dataset_name == "MS_MARCO":
                    ax.text(
                        cur_pos,
                        3.8,
                        f"{latency:.1f}$\\times$",
                        ha="center",
                        va="bottom",
                        fontsize=17,
                        color="black",
                        rotation=90,
                    )
                elif design_name == "NDP-Base" and (dataset_name == "GIST" or dataset_name == "Wiki"):
                    pass
                else:
                    ax.text(
                        cur_pos,
                        4.5,
                        f"{latency:.1f}$\\times$",
                        ha="center",
                        va="bottom",
                        fontsize=17,
                        color="black",
                        rotation=90,
                    )

            bottom = 0.0
            for ratio, component in zip(
                [
                    nbr_fetch_latency_ratio,
                    result_collection_latency_ratio,
                    dist_cal_latency_ratio,
                ],
                [
                    "nbr_fetch_latency_ratio",
                    "result_collection_latency_ratio",
                    "dist_cal_latency_ratio",
                ],
            ):
                height = latency * ratio
                show_label = dataset_name == "GeoMean" and design_name == "ANSMET"
                bar_kw = {
                    "bottom": bottom,
                    "width": bar_width,
                    "color": latency_colors[component],
                    "edgecolor": "black",
                    "linewidth": 0.5,
                }
                if show_label:
                    bar_kw["label"] = component
                ax.bar(cur_pos, height, **bar_kw)
                bottom += height

    ax.set_ylim(0, 6.5)
    ax.set_xlim(-0.2, len(dorder) - 0.2)
    ax.legend(
        labels=["Neighbor List Fetch", "Partial Result Processing", "Distance Calculation"],
        loc="upper center",
        bbox_to_anchor=(0.45, 1.25),
        ncol=4,
        handlelength=1,
        handletextpad=0.1,
        labelspacing=0.1,
        borderpad=0.1,
        columnspacing=0.1,
        fontsize=22,
        frameon=False,
    )

    dataset_labels = ["GeoMean" if n == "GeoMean" else n for n in dorder]
    ax.set_xticks([r + bar_width * 1.5 + begin_offset for r in range(len(dorder))])
    ax.tick_params(axis="x", labelbottom=True, bottom=True, direction="out", length=8, width=1, pad=110)
    ax.tick_params(axis="x", length=0)
    ax.set_xticklabels(dataset_labels, fontsize=22, rotation=10)
    ax.tick_params(axis="y", labelsize=24)

    for i, dataset in enumerate(dorder):
        for j, design in enumerate(dlabels):
            if dataset == "Wiki" and design == "ANSMET":
                continue
            ax.text(i + bar_width * j + begin_offset, -0.03, design, rotation=90, ha="center", va="top", fontsize=22)

    ax.set_ylabel("Normalized\nLatency", fontsize=24)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csv_path = os.path.join("../result", "Data", "latency_breakdown.csv")
    df = load_prepare(csv_path, baseline_name="MY")
    plot_latency_breakdown(df, out_path=os.path.join("../Figures", "fig_18.pdf"))


if __name__ == "__main__":
    main()
