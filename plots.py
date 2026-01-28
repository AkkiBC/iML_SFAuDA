import pandas as pd
import matplotlib.pyplot as plt


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

def plot_all_stability_intervals(
    results_path="results/stability_scores.csv",
    output_dir="results",
):
    """
    Generate stability interval plots for all dataset–explainer pairs.
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