import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import annotate_points

def plot_accuracy_vs_worst_case_stability(
    results_path="results/stability_scores.csv",
    output_path="results/accuracy_vs_worst_case_stability.png",
):
    """
    Failure-centric plot:
    Accuracy vs worst-case (minimum) explanation stability.
    """

    df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    explainer_markers = {
        "shap": "o",
        "lime": "s"
    }

    dataset_colors = {
        "iris": "tab:blue",
        "wine": "tab:green"
    }

    used_positions = set()
    for _, row in df.iterrows():
        x = row["mean_accuracy"]
        y = row["min_correlation"]

        ax.scatter(
            x,
            y,
            marker=explainer_markers.get(row["expl_method"], "o"),
            color=dataset_colors.get(row["dataset"], "gray"),
            edgecolor="black",
            s=90,
            alpha=0.85
        )

        if y < 0.8:
            annotate_points(
                ax=ax,
                x=x,
                y=y,
                text=row["augmentation"],
                used_positions=used_positions,
                base_offset=(6, 6),
                fontsize=9
            )


    ax.axhline(
        y=0.8,
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.7
    )

    ax.text(
        0.01, 0.82,
        "Trust threshold",
        color="red",
        fontsize=10,
        transform=ax.get_yaxis_transform()
    )

    explainer_legend = [
        Line2D([0], [0], marker='o', color='black', linestyle='None',
               markersize=8, label='SHAP'),
        Line2D([0], [0], marker='s', color='black', linestyle='None',
               markersize=8, label='LIME')
    ]

    dataset_legend = [
        Line2D([0], [0], marker='o', color='tab:blue', linestyle='None',
               markersize=8, label='Iris'),
        Line2D([0], [0], marker='o', color='tab:green', linestyle='None',
               markersize=8, label='Wine')
    ]

    legend1 = ax.legend(
        handles=explainer_legend,
        title="Explainer",
        loc="lower right",
        frameon=True
    )

    legend2 = ax.legend(
        handles=dataset_legend,
        title="Dataset",
        loc="lower left",
        frameon=True
    )

    ax.add_artist(legend1)

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Worst-Case Explanation Stability (min correlation)")
    ax.set_title("Accuracy vs Worst-Case Explanation Stability")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.05, 1.05)

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_accuracy_vs_mean_stability(
    results_path="results/stability_scores.csv",
    output_path="results/accuracy_vs_mean_stability.png",
):
    """
    Contrast plot:
    Accuracy vs mean explanation stability.
    Shows how averaging can mask instability.
    """

    df = pd.read_csv(results_path)
    df = df.dropna(subset=["mean_correlation"])

    fig, ax = plt.subplots(figsize=(8, 6))

    explainer_markers = {
        "shap": "o",
        "lime": "s"
    }

    dataset_colors = {
        "iris": "tab:blue",
        "wine": "tab:green"
    }

    used_positions = set()
    for _, row in df.iterrows():
        x = row["mean_accuracy"]
        y = row["mean_correlation"]

        ax.scatter(
            x,
            y,
            marker=explainer_markers.get(row["expl_method"], "o"),
            color=dataset_colors.get(row["dataset"], "gray"),
            edgecolor="black",
            s=90,
            alpha=0.9,
            zorder=3
        )

        annotate_points(
            ax=ax,
            x=x,
            y=y,
            text=row["augmentation"],
            used_positions=used_positions,
            base_offset=(6, 6),
            fontsize=8
        )

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Mean Explanation Stability (correlation)")
    ax.set_title("Accuracy vs Mean Explanation Stability")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.05)

    ax.grid(True, linestyle="--", alpha=0.5)

    explainer_legend = [
        Line2D([0], [0], marker='o', color='black', linestyle='None',
               markersize=8, label='SHAP'),
        Line2D([0], [0], marker='s', color='black', linestyle='None',
               markersize=8, label='LIME')
    ]

    dataset_legend = [
        Line2D([0], [0], marker='o', color='tab:blue', linestyle='None',
               markersize=8, label='Iris'),
        Line2D([0], [0], marker='o', color='tab:green', linestyle='None',
               markersize=8, label='Wine')
    ]

    legend1 = ax.legend(
        handles=explainer_legend,
        title="Explainer",
        loc="lower right",
        frameon=True
    )

    legend2 = ax.legend(
        handles=dataset_legend,
        title="Dataset",
        loc="lower left",
        frameon=True
    )

    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()