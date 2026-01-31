import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────
#  At the top of the file (after imports)
# ────────────────────────────────────────────────

AUGMENTATION_ORDER = [
    'no_augmentation',
    'add_random_noise',
    'feature_jitter',
    'geometric_transform',
    'scale_standard',
    'scale_minmax',
    'scale_robust',
]

# Optional: nicer display names (for labels only)
AUGMENTATION_DISPLAY = {
    'no_augmentation': 'None',
    'add_random_noise': 'Add Noise',
    'feature_jitter': 'Feature Jitter',
    'geometric_transform': 'Geometric',
    'scale_standard': 'Standard Scale',
    'scale_minmax': 'MinMax Scale',
    'scale_robust': 'Robust Scale',
}


# Retain the existing functions for stability intervals
def plot_stability_interval_single(
    df,
    dataset,
    explainer,
    output_path,
    trust_threshold=0.8,
):
    """
    Plot mean explanation stability (point) with worst-case deviation (line)
    for a single (dataset, explainer) pair.
    """

    sub = df[
        (df["dataset"] == dataset) &
        (df["expl_method"] == explainer)
    ].reset_index(drop=True)

    if sub.empty:
        print(f"Skipping empty plot: {dataset} / {explainer}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    x_positions = range(len(sub))

    color = "tab:blue" if dataset == "iris" else "tab:green"

    for i, row in sub.iterrows():
        y_mean = row["mean_correlation"]
        y_min = row["min_correlation"]

        # Mean stability point
        ax.scatter(
            i,
            y_mean,
            color=color,
            edgecolor="black",
            s=90,
            zorder=3
        )

        # Worst-case deviation (vertical line)
        ax.vlines(
            i,
            ymin=y_min,
            ymax=y_mean,
            linewidth=3,
            alpha=0.7,
            zorder=2
        )

    # Trust threshold
    ax.axhline(
        trust_threshold,
        linestyle="--",
        color="red",
        alpha=0.6,
        label="Trust threshold"
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(sub["augmentation"], rotation=30, ha="right")

    ax.set_ylabel("Explanation Stability")
    ax.set_xlabel("Augmentation Strategy")

    ax.set_title(f"{dataset.capitalize()} / {explainer.upper()}")

    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_stability_interval_mnist(
    df,
    output_path,
    trust_threshold=0.8,
):
    """
    Plot MNIST stability with SHAP blocks first, then LIME blocks
    in a single plot.
    """

    sub = df[df["dataset"] == "mnist"].reset_index(drop=True)

    if sub.empty:
        print("Skipping MNIST plot (no data)")
        return

    # Split SHAP / LIME
    shap_df = sub[sub["expl_method"] == "shap"].reset_index(drop=True)
    lime_df = sub[sub["expl_method"] == "lime"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = 0
    xticks = []
    xticklabels = []

    # ---------- SHAP ----------
    for _, row in shap_df.iterrows():
        ax.scatter(
            x,
            row["mean_correlation"],
            color="tab:blue",
            edgecolor="black",
            s=90,
            zorder=3,
            label="SHAP" if x == 0 else None,
        )
        ax.vlines(
            x,
            row["min_correlation"],
            row["mean_correlation"],
            linewidth=3,
            alpha=0.7,
            zorder=2,
            color="tab:blue",
        )
        xticks.append(x)
        xticklabels.append(f"SHAP\n{row['augmentation']}")
        x += 1

    # Gap between SHAP and LIME
    x += 1

    # ---------- LIME ----------
    for _, row in lime_df.iterrows():
        ax.scatter(
            x,
            row["mean_correlation"],
            color="tab:green",
            edgecolor="black",
            s=90,
            zorder=3,
            label="LIME" if "LIME" not in ax.get_legend_handles_labels()[1] else None,
        )
        ax.vlines(
            x,
            row["min_correlation"],
            row["mean_correlation"],
            linewidth=3,
            alpha=0.7,
            zorder=2,
            color="tab:green",
        )
        xticks.append(x)
        xticklabels.append(f"LIME\n{row['augmentation']}")
        x += 1

    # Trust threshold
    ax.axhline(
        trust_threshold,
        linestyle="--",
        color="red",
        alpha=0.6,
        label="Trust threshold",
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_ylabel("Explanation Stability")
    ax.set_title("MNIST – Explanation Stability (SHAP vs LIME)")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# New plotting functions to better highlight accuracy vs. stability trade-offs

def plot_accuracy_vs_stability_scatter(
    df,
    stability_metric="min_correlation",
    output_path="results/accuracy_vs_stability_scatter.png",
    trust_threshold=0.8,
):
    """
    Scatter plot of mean accuracy vs. stability (mean or min correlation).
    Points colored by dataset, shaped by explainer, labeled by augmentation.
    Highlights cases where accuracy is high but stability is low.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors and markers
    dataset_colors = {
        "iris": "tab:blue",
        "wine": "tab:green",
        "mnist": "tab:orange",
    }
    explainer_markers = {
        "shap": "o",
        "lime": "s",
    }

    used_positions = set()  # To avoid overlapping annotations if needed

    for _, row in df.iterrows():
        x = row["mean_accuracy"]
        y = row[stability_metric]  # Use 'mean_correlation' or 'min_correlation'

        ax.scatter(
            x,
            y,
            color=dataset_colors.get(row["dataset"], "gray"),
            marker=explainer_markers.get(row["expl_method"], "o"),
            edgecolor="black",
            s=90,
            alpha=0.85,
            zorder=3,
        )

        # Annotate with augmentation if stability is low
        if y < trust_threshold:
            ax.annotate(
                row["augmentation"],
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", alpha=0.5),
            )

    # Trust threshold horizontal line
    ax.axhline(
        y=trust_threshold,
        linestyle="--",
        color="red",
        alpha=0.7,
        label="Trust Threshold",
    )

    # Legends
    from matplotlib.lines import Line2D

    dataset_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=d)
        for d, c in dataset_colors.items()
    ]
    explainer_legend = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", markersize=8, label=e)
        for e, m in explainer_markers.items()
    ]

    ax.legend(handles=dataset_legend, title="Dataset", loc="upper left")
    ax.add_artist(ax.legend(handles=explainer_legend, title="Explainer", loc="upper right"))

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel(f"Explanation Stability ({stability_metric.replace('_', ' ').title()})")
    ax.set_title("Accuracy vs. Explanation Stability Across Configurations")
    ax.set_xlim(0.3, 1.05)  # Based on your data range
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_dual_axis_bar(
    df,
    dataset,
    explainer,
    output_path,
    trust_threshold=0.8,
):
    sub = df[
        (df["dataset"] == dataset) &
        (df["expl_method"] == explainer)
    ]

    if sub.empty:
        print(f"Skipping empty plot: {dataset} / {explainer}")
        return

    # Sort using the defined order (missing augs are ignored)
    sub = sub.copy()
    sub['aug_sort'] = sub['augmentation'].map(
        {aug: i for i, aug in enumerate(AUGMENTATION_ORDER)}
    )
    sub = sub.sort_values('aug_sort').drop(columns='aug_sort').reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))  # slightly wider

    # Bars: accuracy
    bars = ax1.bar(
        range(len(sub)),
        sub["mean_accuracy"],
        color="tab:blue",
        alpha=0.75,
        label="Mean Accuracy",
        zorder=3,
    )
    ax1.set_ylabel("Mean Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.05)

    # Set x-ticks early
    ax1.set_xticks(range(len(sub)))
    ax1.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(a, a.replace('_', ' ').title()) for a in sub["augmentation"]],
        rotation=40,
        ha="right",
    )

    # Line + error for stability (right axis)
    ax2 = ax1.twinx()
    ax2.errorbar(
        range(len(sub)),
        sub["mean_correlation"],
        yerr=(sub["mean_correlation"] - sub["min_correlation"], [0]*len(sub)),
        fmt="o-",
        color="tab:red",
        linewidth=2.2,
        markersize=8,
        capsize=6,
        label="Stability (Mean with Min Error)",
        zorder=4,
    )
    ax2.set_ylabel("Explanation Stability", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.4, 1.05)

    # Trust threshold
    ax2.axhline(trust_threshold, linestyle="--", color="gray", alpha=0.7, label="Trust Threshold")

    ax1.set_xlabel("Augmentation Strategy")
    ax1.set_title(f"{dataset.capitalize()} / {explainer.upper()}: Accuracy vs. Stability")

    # Legend – move to lower right (or try 'upper center' / 'lower center')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, 
               loc="lower left", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_stability_heatmap(
    df,
    output_path="results/stability_heatmap.png",
):
    # Define row order: group by dataset, SHAP before LIME, iris/wine before mnist
    row_order = []
    for ds in ['iris', 'wine', 'mnist']:
        for expl in ['shap', 'lime']:
            if ((df["dataset"] == ds) & (df["expl_method"] == expl)).any():
                row_order.append(f"{ds}-{expl}")

    # Column order using AUGMENTATION_ORDER
    col_order = [a for a in AUGMENTATION_ORDER if a in df["augmentation"].unique()]

    # Pivot min_correlation
    pivot = df.pivot_table(
        index=["dataset", "expl_method"],
        columns="augmentation",
        values="min_correlation",
        aggfunc="first"   # in case of duplicates
    ).reindex(index=pd.MultiIndex.from_tuples(
        [(ds, expl) for ds, expl in [r.split('-') for r in row_order]],
        names=["dataset", "expl_method"]
    ), columns=col_order).fillna(0)

    # Pivot for accuracy annotations
    acc_pivot = df.pivot_table(
        index=["dataset", "expl_method"],
        columns="augmentation",
        values="mean_accuracy",
        aggfunc="first"
    ).reindex_like(pivot).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        pivot,
        annot=acc_pivot.map(lambda x: f"{x:.2f}" if x > 0 else ""),
        fmt="",
        cmap="RdYlGn_r",          # green=high stability, red=low
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Min Correlation (Worst-case Stability)"},
        annot_kws={"size": 9, "weight": "bold"},
    )

    # Improve labels
    ax.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(t.get_text(), t.get_text().replace('_', ' ').title())
         for t in ax.get_xticklabels()],
        rotation=45, ha="right"
    )

    # Nicer y-labels
    yticklabels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        ds, expl = text.split('-') if '-' in text else (text, '')
        nice = f"{ds.capitalize()} / {expl.upper()}"
        yticklabels.append(nice)
    ax.set_yticklabels(yticklabels, rotation=0)

    ax.set_title("Explanation Stability Heatmap\n(Annotations = Mean Accuracy)", fontsize=13, pad=15)
    ax.set_xlabel("Augmentation Strategy", fontsize=11)
    ax.set_ylabel("Dataset / Explainer", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_all_stability_intervals(
    results_path="results/stability_scores.csv",
    output_dir="results",
):
    """
    Generate stability interval plots for all dataset–explainer pairs.
    (Retained from original)
    """

    df = pd.read_csv(results_path)
    # Exclude scaling-based augmentations for LIME (not meaningful)
    df = df[
        ~(
            (df["expl_method"] == "lime") &
            (df["augmentation"].isin([
                "scale_standard",
                "scale_minmax",
                "scale_robust",
            ]))
        )
    ]
    df = df.dropna(subset=["mean_correlation", "min_correlation"])

    configs = [
        ("iris", "shap"),
        ("wine", "shap"),
        ("iris", "lime"),
        ("wine", "lime"),
    ]


    for dataset, explainer in configs:
        output_path = f"{output_dir}/stability_interval_{dataset}_{explainer}.png"

        plot_stability_interval_single(
            df=df,
            dataset=dataset,
            explainer=explainer,
            output_path=output_path
        )

    plot_stability_interval_mnist(
        df=df,
        output_path=f"{output_dir}/stability_interval_mnist.png"
    )

def generate_all_plots(
    results_path="results/stability_scores.csv",
    output_dir="results",
):
    """
    Wrapper to generate all proposed plots.
    """
    df = pd.read_csv(results_path)
    df = df.dropna(subset=["mean_correlation", "min_correlation"])

    # Existing stability intervals
    plot_all_stability_intervals(results_path=results_path, output_dir=output_dir)

    # New plots
    plot_accuracy_vs_stability_scatter(df, stability_metric="min_correlation", output_path=f"{output_dir}/accuracy_vs_min_stability.png")
    plot_accuracy_vs_stability_scatter(df, stability_metric="mean_correlation", output_path=f"{output_dir}/accuracy_vs_mean_stability.png")

    # Dual-axis for each config
    configs = df[["dataset", "expl_method"]].drop_duplicates().values
    for dataset, explainer in configs:
        plot_dual_axis_bar(
            df,
            dataset,
            explainer,
            output_path=f"{output_dir}/dual_axis_{dataset}_{explainer}.png"
        )

    # Heatmap
    plot_stability_heatmap(df, output_path=f"{output_dir}/stability_heatmap.png")

if __name__ == "__main__":
    generate_all_plots()