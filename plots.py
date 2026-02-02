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

# Nicer labels for plots (display only)
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
# STABILITY HEATMAP
# =========================

def plot_stability_heatmap(
    df,
    output_path="results/stability_heatmap_min.png",
    value_col="min_correlation",          # or "mean_correlation"
):
    # Row order: iris/wine/mnist, SHAP before LIME
    row_order = []
    for ds in ['iris', 'wine', 'mnist']:
        for expl in ['shap', 'lime']:
            if ((df["dataset"] == ds) & (df["expl_method"] == expl)).any():
                row_order.append(f"{ds}-{expl}")

    col_order = [a for a in AUGMENTATION_ORDER if a in df["augmentation"].unique()]

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

    # Gray out missing cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pd.isna(pivot.iloc[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lightgray', lw=0, zorder=10))

    ax.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(t.get_text(), t.get_text().replace('_', ' ').title())
         for t in ax.get_xticklabels()],
        rotation=40, ha="right", fontsize=10
    )

    yticklabels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        if '-' in text:
            ds, expl = text.split('-')
            nice = f"{ds.capitalize()} / {expl.upper()}"
            yticklabels.append(nice)
        else:
            yticklabels.append(text)
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
# DUAL-AXIS PLOT (accuracy + stability per dataset)
# =========================

def plot_dual_axis_combined(
    df,
    dataset,
    output_path,
    figsize=(9.5, 5.6),
):
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        print(f"Skipping empty plot: {dataset}")
        return

    ordered_augs = [a for a in AUGMENTATION_ORDER if a in sub["augmentation"].unique()]
    if not ordered_augs:
        print(f"No matching augmentations for {dataset}")
        return

    acc_df = sub.groupby("augmentation")["mean_accuracy"].first().reindex(ordered_augs)

    shap_mean = sub[sub["expl_method"] == "shap"].set_index("augmentation")["mean_correlation"].reindex(ordered_augs)
    shap_min  = sub[sub["expl_method"] == "shap"].set_index("augmentation")["min_correlation"].reindex(ordered_augs)
    lime_mean = sub[sub["expl_method"] == "lime"].set_index("augmentation")["mean_correlation"].reindex(ordered_augs)
    lime_min  = sub[sub["expl_method"] == "lime"].set_index("augmentation")["min_correlation"].reindex(ordered_augs)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Accuracy bars
    x = np.arange(len(ordered_augs))
    bar_width = 0.58

    valid_acc = acc_df.dropna()
    valid_x_acc = [ordered_augs.index(a) for a in valid_acc.index]

    ax1.bar(
        valid_x_acc,
        valid_acc.values,
        bar_width,
        color="lightgray",
        edgecolor="black",
        label="Mean Accuracy",
        zorder=1
    )

    ax1.set_ylabel("Mean Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [AUGMENTATION_DISPLAY.get(a, a.replace("_", " ").title()) for a in ordered_augs],
        rotation=42, ha="right", fontsize=10
    )
    ax1.set_xlabel("Augmentation Strategy", fontsize=11)

    # Stability twin axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explanation Stability (Correlation)", fontsize=11)
    ax2.set_ylim(-0.4, 1.05)

    # SHAP
    valid_shap_mean = shap_mean.dropna()
    if not valid_shap_mean.empty:
        shap_x = np.array([ordered_augs.index(a) for a in valid_shap_mean.index])
        ax2.plot(shap_x, valid_shap_mean, color="navy", marker="o", ms=8, mec="black", mew=0.8,
                 linestyle="-", linewidth=2.4, label="SHAP Correlation", zorder=5)

        for i, aug in enumerate(valid_shap_mean.index):
            xi = ordered_augs.index(aug)
            y_mean = valid_shap_mean[aug]
            y_min = shap_min[aug]
            if not pd.isna(y_min):
                ax2.vlines(xi, y_min, y_mean, color="navy", lw=3.0, alpha=0.70, zorder=3)
                ax2.hlines(y_min, xi-0.07, xi+0.07, color="navy", lw=2.4, alpha=0.70)

    # LIME
    valid_lime_mean = lime_mean.dropna()
    if not valid_lime_mean.empty:
        lime_x = np.array([ordered_augs.index(a) for a in valid_lime_mean.index])
        ax2.plot(lime_x, valid_lime_mean, color="forestgreen", marker="o", ms=8, mec="black", mew=0.8,
                 linestyle="-", linewidth=2.4, label="LIME Correlation", zorder=5)

        for i, aug in enumerate(valid_lime_mean.index):
            xi = ordered_augs.index(aug)
            y_mean = valid_lime_mean[aug]
            y_min = lime_min[aug]
            if not pd.isna(y_min):
                ax2.vlines(xi, y_min, y_mean, color="forestgreen", lw=3.0, alpha=0.70, zorder=3)
                ax2.hlines(y_min, xi-0.07, xi+0.07, color="forestgreen", lw=2.4, alpha=0.70)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=True,
        edgecolor="lightgray",
        borderpad=0.7
    )

    title_text = f"{dataset.capitalize()} – Accuracy and Explanation Stability\n"
    ax1.set_title(title_text, fontsize=13, pad=10)

    ax1.grid(True, linestyle="--", alpha=0.30, zorder=0)

    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



# =========================
# GENERATE ALL PLOTS
# =========================

def generate_all_plots(
    results_path="results/stability_scores.csv",
    output_dir="results",
):
    df = pd.read_csv(results_path)
    df = df.dropna(subset=["mean_correlation", "min_correlation"], how="all")

    # Dual-axis plots per dataset
    datasets = sorted(df["dataset"].unique())
    for ds in datasets:
        plot_dual_axis_combined(
            df,
            ds,
            f"{output_dir}/dual_axis_combined_{ds}.png"
        )

    # Heatmaps
    plot_stability_heatmap(df, f"{output_dir}/stability_heatmap_min.png",   value_col="min_correlation")
    plot_stability_heatmap(df, f"{output_dir}/stability_heatmap_mean.png", value_col="mean_correlation")

if __name__ == "__main__":
    generate_all_plots()