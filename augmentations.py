import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def no_augmentation(X, dataset_type=None):
    return X

def add_random_noise(X, noise_level=0.1, dataset_type=None):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def geometric_transform(X, dataset_type='tabular'):
    if dataset_type == 'image':
        # Simple shift for flattened MNIST (reshape, shift, flatten back)
        X_reshaped = X.reshape(-1, 28, 28)
        shifts = np.random.randint(-2, 3, size=(len(X_reshaped), 2))
        for i in range(len(X_reshaped)):
            X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][0], axis=0)
            X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][1], axis=1)
        return X_reshaped.reshape(-1, 28*28)
    else:
        # For tabular, fallback to noise
        return add_random_noise(X)
    
def scale_standard(X, dataset_type=None):
    # Standard scaling: zero mean, unit variance
    if dataset_type != 'tabular':
        return X
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def scale_minmax(X, dataset_type=None):
    # Min–max scaling to [0, 1]
    if dataset_type != 'tabular':
        return X
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def scale_robust(X, dataset_type=None):
    # Robust scaling using median and IQR.
    # Representation-preserving for tabular data.
    if dataset_type != 'tabular':
        return X
    scaler = RobustScaler()
    return scaler.fit_transform(X)

def feature_jitter(X, noise_ratio=0.05, dataset_type=None):
    # Add Gaussian noise proportional to per-feature standard deviation.
    # Preserves relative feature importance.
    if dataset_type != 'tabular':
        return X

    std = X.std(axis=0, keepdims=True)
    noise = np.random.normal(0, noise_ratio * std, size=X.shape)
    return X + noise

def bootstrap_resample(X, dataset_type=None):
    # Bootstrap resampling of rows (with replacement).
    # Preserves feature space, changes empirical distribution.
    idx = np.random.choice(len(X), size=len(X), replace=True)
    return X[idx]
