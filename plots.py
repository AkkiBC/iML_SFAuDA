import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =========================
# AUGMENTATION ORDER & DISPLAY NAMES
# =========================

AUGMENTATION_ORDER = [
    'no_augmentation',
    'add_random_noise',
    'geometric_transform',
    'feature_jitter',
    'scale_standard',
    'scale_minmax',
    'scale_robust',
]

AUGMENTATION_DISPLAY = {
    'no_augmentation': 'None',
    'add_random_noise': 'Add Noise',
    'feature_jitter': 'Feature Jitter',
    'geometric_transform': 'Geometric',
    'scale_standard': 'Standard Scale',
    'scale_minmax': 'MinMax Scale',
    'scale_robust': 'Robust Scale',
}


# =========================
# SINGLE DATASET INTERVAL PLOT
# =========================

def plot_stability_interval_single(df, dataset, explainer, output_path, trust_threshold=0.8):
    sub = df[(df["dataset"] == dataset) & (df["expl_method"] == explainer)].reset_index(drop=True)
    
    if sub.empty:
        print(f"Skipping empty plot: {dataset} / {explainer}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    x_positions = range(len(sub))
    color = "tab:blue" if dataset == "iris" else "tab:green"

    for i, row in sub.iterrows():
        ax.scatter(i, row["mean_correlation"], color=color, edgecolor="black", s=90, zorder=3)
        ax.vlines(i, row["min_correlation"], row["mean_correlation"], linewidth=3, alpha=0.7, zorder=2)

    ax.axhline(trust_threshold, linestyle="--", color="red", alpha=0.6, label="Trust threshold")
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


# =========================
# MNIST GROUPED INTERVAL PLOT
# =========================

def plot_stability_interval_mnist(df, output_path, trust_threshold=0.8):
    sub = df[df["dataset"] == "mnist"].reset_index(drop=True)
    if sub.empty:
        print("Skipping MNIST plot (no data)")
        return

    shap_df = sub[sub["expl_method"] == "shap"].reset_index(drop=True)
    lime_df = sub[sub["expl_method"] == "lime"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = 0
    xticks = []
    xticklabels = []

    # SHAP group
    for _, row in shap_df.iterrows():
        ax.scatter(x, row["mean_correlation"], color="tab:blue", edgecolor="black", s=90, zorder=3,
                   label="SHAP" if x == 0 else None)
        ax.vlines(x, row["min_correlation"], row["mean_correlation"], linewidth=3, alpha=0.7, zorder=2,
                  color="tab:blue")
        xticks.append(x)
        xticklabels.append(f"SHAP\n{row['augmentation']}")
        x += 1

    x += 1  # gap between groups

    # LIME group
    for _, row in lime_df.iterrows():
        ax.scatter(x, row["mean_correlation"], color="tab:green", edgecolor="black", s=90, zorder=3,
                   label="LIME" if "LIME" not in ax.get_legend_handles_labels()[1] else None)
        ax.vlines(x, row["min_correlation"], row["mean_correlation"], linewidth=3, alpha=0.9, zorder=2,
                  color="tab:green")
        xticks.append(x)
        xticklabels.append(f"LIME\n{row['augmentation']}")
        x += 1

    ax.axhline(trust_threshold, linestyle="--", color="red", alpha=0.9, label="Trust threshold")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_ylabel("Explanation Stability")
    ax.set_xlabel("Augmentation Strategy (Grouped by Explainer)")
    ax.set_title("MNIST Explanation Stability")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================
# STABILITY HEATMAP
# =========================

def plot_stability_heatmap(df, output_path="results/stability_heatmap_min.png", value_col="min_correlation"):
    # Row order: iris/wine/mnist → SHAP then LIME
    row_order = []
    for ds in ['iris', 'wine', 'mnist']:
        for expl in ['shap', 'lime']:
            if ((df["dataset"] == ds) & (df["expl_method"] == expl)).any():
                row_order.append(f"{ds}-{expl}")

    col_order = [a for a in AUGMENTATION_ORDER if a in df["augmentation"].unique()]

    # Pivot
    pivot = df.pivot_table(
        index=["dataset", "expl_method"],
        columns="augmentation",
        values=value_col,
        aggfunc="first"
    ).reindex(
        index=pd.MultiIndex.from_tuples(
            [(ds, expl) for ds, expl in [r.split('-') for r in row_order]],
            names=["dataset", "expl_method"]
        ),
        columns=col_order
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=-0.1,
        vmax=1.0,
        center=0.5,
        annot=pivot.map(lambda x: f"{x:.3f}" if pd.notna(x) else ""),
        fmt="",
        linewidths=0.9,
        linecolor="white",
        cbar_kws={"label": f"{value_col.replace('_', ' ').title()}"},
        ax=ax,
        annot_kws={"size": 9.5, "weight": "bold"},
    )

    # Gray out missing values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pd.isna(pivot.iloc[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lightgray', lw=0, zorder=10))

    # Nice labels
    ax.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(t.get_text(), t.get_text().replace('_', ' ').title())
         for t in ax.get_xticklabels()],
        rotation=40, ha="right", fontsize=10
    )

    yticklabels = [f"{ds.capitalize()} / {expl.upper()}" for text in ax.get_yticklabels()
                   for ds, expl in [text.get_text().split('-')]]
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=10.5)

    ax.set_title(
        f"Stability of SHAP and LIME Explanations\n"
        f"({value_col.replace('_', ' ').title()} across Augmentation Strategies)",
        fontsize=14, pad=20
    )
    ax.set_xlabel("Augmentation Strategy", fontsize=11)
    ax.set_ylabel("Dataset / Explainer", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# DUAL AXIS — ACCURACY + STABILITY
# =========================

def plot_dual_axis_combined(df, dataset, output_path, figsize=(9.5, 5.6)):
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return

    ordered_augs = [a for a in AUGMENTATION_ORDER if a in sub["augmentation"].unique()]
    if not ordered_augs:
        return

    acc_df = sub.groupby("augmentation")["mean_accuracy"].first().reindex(ordered_augs)

    shap_mean = sub[sub["expl_method"] == "shap"].set_index("augmentation")["mean_correlation"].reindex(ordered_augs)
    shap_min  = sub[sub["expl_method"] == "shap"].set_index("augmentation")["min_correlation"].reindex(ordered_augs)
    lime_mean = sub[sub["expl_method"] == "lime"].set_index("augmentation")["mean_correlation"].reindex(ordered_augs)
    lime_min  = sub[sub["expl_method"] == "lime"].set_index("augmentation")["min_correlation"].reindex(ordered_augs)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Accuracy bars
    x = np.arange(len(ordered_augs))
    ax1.bar(x, acc_df, 0.58, color="lightgray", edgecolor="black", label="Mean Accuracy", zorder=1)
    ax1.set_ylabel("Mean Accuracy")
    ax1.set_ylim(0, 1.10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(a, a.replace("_", " ").title()) for a in ordered_augs],
        rotation=42, ha="right"
    )
    ax1.set_xlabel("Augmentation Strategy")

    # Stability (twin axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explanation Stability (Correlation)")
    ax2.set_ylim(-0.4, 1.05)

    # SHAP
    valid_shap = shap_mean.dropna()
    if not valid_shap.empty:
        xs = np.array([ordered_augs.index(a) for a in valid_shap.index])
        ax2.plot(xs, valid_shap, color="navy", marker="o", ms=8, mec="black", mew=0.8,
                 linewidth=2.4, label="SHAP Correlation", zorder=5)

        for i, aug in enumerate(valid_shap.index):
            xi = ordered_augs.index(aug)
            ax2.vlines(xi, shap_min[aug], valid_shap[aug], color="navy", lw=3, alpha=0.7)

    # LIME
    valid_lime = lime_mean.dropna()
    if not valid_lime.empty:
        xs = np.array([ordered_augs.index(a) for a in valid_lime.index])
        ax2.plot(xs, valid_lime, color="forestgreen", marker="o", ms=8, mec="black", mew=0.8,
                 linewidth=2.4, label="LIME Correlation", zorder=5)

        for i, aug in enumerate(valid_lime.index):
            xi = ordered_augs.index(aug)
            ax2.vlines(xi, lime_min[aug], valid_lime[aug], color="forestgreen", lw=3, alpha=0.7)

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3)

    ax1.set_title(f"{dataset.capitalize()} – Accuracy and Explanation Stability")
    ax1.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# ACCURACY vs STABILITY SCATTER
# =========================

def plot_accuracy_vs_stability_scatter(df, stability_metric="min_correlation",
                                      output_path="results/accuracy_vs_stability_scatter.png",
                                      trust_threshold=0.8):
    fig, ax = plt.subplots(figsize=(8, 6))

    dataset_colors = {"iris": "tab:blue", "wine": "tab:green", "mnist": "tab:orange"}
    explainer_markers = {"shap": "o", "lime": "s"}

    for _, row in df.iterrows():
        ax.scatter(
            row["mean_accuracy"],
            row[stability_metric],
            color=dataset_colors.get(row["dataset"], "gray"),
            marker=explainer_markers.get(row["expl_method"], "o"),
            edgecolor="black",
            s=90,
            alpha=0.85,
            zorder=3
        )

        if row[stability_metric] < trust_threshold:
            ax.annotate(
                row["augmentation"],
                (row["mean_accuracy"], row[stability_metric]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", alpha=0.5)
            )

    ax.axhline(trust_threshold, linestyle="--", color="red", alpha=0.7, label="Trust Threshold")

    from matplotlib.lines import Line2D
    dataset_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=d)
                      for d, c in dataset_colors.items()]
    explainer_legend = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", markersize=8, label=e)
                        for e, m in explainer_markers.items()]

    ax.legend(handles=dataset_legend, title="Dataset", loc="upper left")
    ax.add_artist(ax.legend(handles=explainer_legend, title="Explainer", loc="upper right"))

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel(f"Explanation Stability ({stability_metric.replace('_', ' ').title()})")
    ax.set_title("Accuracy vs. Explanation Stability Across Configurations")
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================
# GENERATE ALL PLOTS
# =========================

def generate_all_plots(results_path="results/stability_scores.csv", output_dir="results"):
    df = pd.read_csv(results_path)
    df = df.dropna(subset=["mean_correlation", "min_correlation"], how="all")

    # Interval plots (single + mnist grouped)
    configs = [("iris", "shap"), ("wine", "shap"), ("iris", "lime"), ("wine", "lime")]
    for dataset, explainer in configs:
        plot_stability_interval_single(
            df, dataset, explainer,
            f"{output_dir}/stability_interval_{dataset}_{explainer}.png"
        )

    plot_stability_interval_mnist(
        df, f"{output_dir}/stability_interval_mnist.png"
    )

    # Dual axis per dataset
    for ds in sorted(df["dataset"].unique()):
        plot_dual_axis_combined(
            df, ds, f"{output_dir}/dual_axis_combined_{ds}.png"
        )

    # Heatmaps — both mean and min
    plot_stability_heatmap(df, f"{output_dir}/stability_heatmap_min.png",  value_col="min_correlation")
    plot_stability_heatmap(df, f"{output_dir}/stability_heatmap_mean.png", value_col="mean_correlation")

    # Final summary scatter
    plot_accuracy_vs_stability_scatter(
        df,
        stability_metric="min_correlation",
        output_path=f"{output_dir}/accuracy_vs_stability_scatter_min.png"
    )


if __name__ == "__main__":
    generate_all_plots()