from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main() -> None:
    overall_df = pd.read_csv("../result/Data/overall.csv")
    overall_df["QPS"] = pd.to_numeric(
        overall_df["QPS"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    selected_designs = ["SCANN", "PIMANN", "ANNA", "DF-GAS", "ANSMET", "NasZip"]
    available_designs = [d for d in selected_designs if d in overall_df["design_name"].unique()]
    df_sel = overall_df[overall_df["design_name"].isin(available_designs)].copy()

    cpu_qps = overall_df[overall_df["design_name"] == "CPU"][["dataset", "QPS"]].set_index("dataset")[
        "QPS"
    ]
    df_sel["QPS_norm"] = df_sel.apply(lambda row: row["QPS"] / cpu_qps[row["dataset"]], axis=1)

    geo = (
        df_sel.groupby("design_name")["QPS_norm"]
        .apply(lambda s: np.exp(np.mean(np.log(s))))
        .reset_index()
    )
    geo["dataset"] = "GeoMean"

    plot_df = pd.concat([df_sel, geo], ignore_index=True)

    dataset_order = ["SIFT", "GIST", "GloVe", "BigANN", "Wiki", "MS_MARCO", "GeoMean"]

    color_palette = {
        "NDP-Baseline": "#E0CCE0",
        "NDSEARCH": "#FAECB2",
        "ANSMET": "#BEDEF4",
        "ANNA": "#E5C9B1",
        "CPU": "#E0CCE0",
        "DF-GAS": "#F6D965",
        "PIMANN": "#a9dfbf",
        "SCANN": "#d7bde2",
        "NasZip": "#75ABD0",
    }

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(13, 3.4))
    sns.barplot(
        x="dataset",
        y="QPS_norm",
        hue="design_name",
        data=plot_df,
        order=dataset_order,
        hue_order=available_designs,
        palette=color_palette,
        edgecolor="black",
        linewidth=1,
        alpha=0.9,
        ax=ax,
    )

    ax.set_ylabel("Normalized Speedup", fontsize=16)
    ax.set_xlabel("Datasets", fontsize=17)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 36)

    for j, container in enumerate(ax.containers):
        design_name = available_designs[j]
        for _, (bar, dataset) in enumerate(zip(container, dataset_order)):
            height = bar.get_height()
            if height <= 0:
                continue
            label_va = "bottom"
            y_offset = 0.2
            x_offset = 0
            if dataset == "GIST" and design_name == "NasZip":
                y_offset = -0.4
                x_offset = 0.15
                label_va = "top"
            x = bar.get_x() + bar.get_width() / 2 + x_offset
            y = height + y_offset
            ax.text(
                x,
                y,
                f"{height:.1f}$\\times$",
                ha="center",
                va=label_va,
                fontsize=16,
                rotation=90,
            )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.67, 1.1),
        fontsize=16,
        frameon=False,
    )

    plt.tight_layout()
    os.makedirs("../Figures", exist_ok=True)
    plt.savefig("../Figures/fig_15.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
