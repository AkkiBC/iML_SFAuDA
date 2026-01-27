import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_vs_worst_case_stability(    
    results_path="results/stability_scores.csv",
    output_path="results/stability_by_augmentation.png",
    ):
    """
    Failure-centric plot:
    Accuracy vs worst-case (minimum) explanation stability.
    """

    df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Marker styles per explainer
    explainer_markers = {
        "shap": "o",
        "lime": "s"
    }

    # Color per dataset
    dataset_colors = {
        "iris": "tab:blue",
        "wine": "tab:green"
    }

    for _, row in df.iterrows():
        ax.scatter(
            row["mean_accuracy"],
            row["min_correlation"],
            marker=explainer_markers.get(row["expl_method"], "o"),
            color=dataset_colors.get(row["dataset"], "gray"),
            edgecolor="black",
            s=90,
            alpha=0.85
        )

        # Label only unstable points (failure cases)
        if row["min_correlation"] < 0.8:
            ax.annotate(
                row["augmentation"],
                (row["mean_accuracy"], row["min_correlation"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9
            )

    # Trust threshold
    ax.axhline(
        y=0.8,
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.7,
        label="Trust threshold"
    )

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Worst-Case Explanation Stability (min correlation)")
    ax.set_title("Accuracy vs Worst-Case Explanation Stability")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.05, 1.05)

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
