import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# --------------------------------------------------
# Baseline
# --------------------------------------------------

def no_augmentation(X, dataset_type=None):
    """
    No data augmentation.
    Serves as reference point for stability.
    """
    return X


# --------------------------------------------------
# Noise-based (representation-perturbing)
# --------------------------------------------------

def add_random_noise(X, noise_level=0.05, dataset_type=None):
    """
    Add isotropic Gaussian noise.
    NOTE: Not representation-preserving, but useful as stress test.
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def feature_jitter(X, noise_ratio=0.05, dataset_type=None):
    """
    Feature-wise Gaussian jitter proportional to feature std.
    Considered a *mild* perturbation for tabular data.
    """
    if dataset_type != "tabular":
        return X

    std = np.std(X, axis=0, keepdims=True)
    noise = np.random.normal(0, noise_ratio * std, size=X.shape)
    return X + noise


# --------------------------------------------------
# Preprocessing / representation-level perturbations
# (Supervisor-approved)
# --------------------------------------------------

def scale_standard(X, dataset_type=None):
    """
    Standard scaling (zero mean, unit variance).
    Representation-preserving preprocessing.
    """
    if dataset_type != "tabular":
        return X

    scaler = StandardScaler()
    return scaler.fit_transform(X)


def scale_minmax(X, dataset_type=None):
    """
    Min–max scaling to [0, 1].
    Representation-preserving preprocessing.
    """
    if dataset_type != "tabular":
        return X

    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def scale_robust(X, dataset_type=None):
    """
    Robust scaling using median and IQR.
    Less sensitive to outliers.
    """
    if dataset_type != "tabular":
        return X

    scaler = RobustScaler()
    return scaler.fit_transform(X)


# --------------------------------------------------
# Image-only (for future MNIST extension)
# --------------------------------------------------

def geometric_transform(X, dataset_type=None):
    """
    Simple geometric shift for images (MNIST only).
    NOT used for tabular data.
    """
    if dataset_type != "image":
        return X

    X_reshaped = X.reshape(-1, 28, 28)
    shifts = np.random.randint(-2, 3, size=(len(X_reshaped), 2))

    for i in range(len(X_reshaped)):
        X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][0], axis=0)
        X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][1], axis=1)

    return X_reshaped.reshape(-1, 28 * 28)
