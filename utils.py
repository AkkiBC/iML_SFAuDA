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
    X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten and normalize
    X_test = X_test.reshape(-1, 28*28) / 255.0
    return X_train[:5000], X_test[:1000], y_train[:5000], y_test[:1000]  # Subset for speed

def save_results(stability_scores, filename='results/stability_scores.csv'):
    pd.DataFrame(stability_scores).to_csv(filename, index=False)