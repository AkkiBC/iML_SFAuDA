import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import pandas as pd

def load_tabular_dataset(name):
    if name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        mask = y < 2  # binary classification
        X, y = X[mask], y[mask]
    elif name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
    else:
        raise ValueError("Unsupported dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_image_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Keep only digits 1 and 4
    train_mask = np.isin(y_train, [1, 4])
    test_mask  = np.isin(y_test,  [1, 4])
    
    X_train = X_train[train_mask].reshape(-1, 28*28) / 255.0
    y_train = y_train[train_mask]
    X_test  = X_test[test_mask].reshape(-1, 28*28)  / 255.0
    y_test  = y_test[test_mask]
    
    # Make binary: 1 → 0, 4 → 1  (or vice versa – doesn't matter)
    y_train = (y_train == 4).astype(int)
    y_test  = (y_test  == 4).astype(int)
    
    # Optional: take smaller subset if still too slow
    # X_train = X_train[:8000]
    # y_train = y_train[:8000]
    
    return X_train[:5000], X_test[:1000], y_train[:5000], y_test[:1000]

def save_results(stability_scores, filename='results/stability_scores.csv'):
    pd.DataFrame(stability_scores).to_csv(filename, index=False)

def make_performance_table(
    input_csv="results/stability_scores.csv",
    output_md="results/table_performance.md",
):
    """
    Table A — Model performance only.
    Columns: Dataset | Augmentation | Explainer | Accuracy
    """

    df = pd.read_csv(input_csv)

    table = (
        df[["dataset", "augmentation", "expl_method", "mean_accuracy"]]
        .rename(columns={
            "dataset": "Dataset",
            "augmentation": "Augmentation",
            "expl_method": "Explainer",
            "mean_accuracy": "Accuracy",
        })
        .drop_duplicates()
        .sort_values(["Dataset", "Explainer", "Augmentation"])
        .round(3)
        .reset_index(drop=True)
    )

    table.to_markdown(output_md, index=False)

    return table


def make_stability_table(
    input_csv="results/stability_scores.csv",
    output_md="results/table_stability.md",
):
    """
    Table B — Explanation stability.
    Columns: Dataset | Augmentation | Explainer | Mean Stability | Worst-Case Stability
    """

    df = pd.read_csv(input_csv)

    table = (
        df[[
            "dataset",
            "augmentation",
            "expl_method",
            "mean_correlation",
            "min_correlation",
        ]]
        .rename(columns={
            "dataset": "Dataset",
            "augmentation": "Augmentation",
            "expl_method": "Explainer",
            "mean_correlation": "Mean Stability",
            "min_correlation": "Worst-Case Stability",
        })
        .dropna(subset=["Mean Stability", "Worst-Case Stability"])
        .drop_duplicates()
        .sort_values(["Dataset", "Explainer", "Augmentation"])
        .round(3)
        .reset_index(drop=True)
    )

    table.to_markdown(output_md, index=False)

    return table