import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_stability(
    results_path="results/stability_scores.csv",
    output_path="results/accuracy_vs_stability.png"
):
    df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(7, 5))

    explainer_markers = {
        "shap": "o",
        "lime": "s"
    }

    augmentations = df["augmentation"].unique()
    colors = plt.cm.tab10(range(len(augmentations)))
    aug_colors = dict(zip(augmentations, colors))

    for expl_method, marker in explainer_markers.items():
        for aug in augmentations:
            subset = df[
                (df["expl_method"] == expl_method) &
                (df["augmentation"] == aug)
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset["mean_accuracy"],
                subset["mean_correlation"],
                marker=marker,
                color=aug_colors[aug],
                edgecolor="black",
                alpha=0.6,
                label=f"{aug}" if expl_method == list(explainer_markers.keys())[0] else None
            )

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Mean Explanation Stability (Correlation)")
    ax.set_title("Accuracy vs Explanation Stability")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles_aug = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", label=aug)
        for aug, color in aug_colors.items()
    ]

    handles_expl = [
        plt.Line2D([0], [0], marker=marker, color="black",
                   linestyle="", label=expl)
        for expl, marker in explainer_markers.items()
    ]

    legend1 = ax.legend(handles=handles_aug, title="Augmentation", loc="lower left")
    legend2 = ax.legend(handles=handles_expl, title="Explainer", loc="lower right")
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stability_by_augmentation(
    results_path="results/stability_scores.csv",
    output_path="results/stability_by_augmentation.png"
):
    df = pd.read_csv(results_path)

    order = [
        "no_augmentation",
        "add_random_noise",
        "scale_standard",
        "scale_minmax"
    ]

    pivot = df.pivot_table(
        index="augmentation",
        columns="expl_method",
        values="mean_correlation"
    )

    pivot = pivot.reindex(order)

    ax = pivot.plot(
        kind="bar",
        figsize=(7, 5),
        edgecolor="black"
    )

    ax.set_ylabel("Mean Explanation Stability (Correlation)")
    ax.set_xlabel("Augmentation")
    ax.set_title("Explanation Stability by Augmentation")

    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Explainer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
