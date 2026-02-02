import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# =========================
# NO AUGMENTATION (BASELINE)
# =========================

def no_augmentation(X, dataset_type=None):
    return X


# =========================
# NOISE-BASED AUGMENTATIONS
# =========================

def add_random_noise(X, noise_level=0.05, dataset_type=None):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def feature_jitter(X, noise_ratio=0.05, dataset_type=None):
    if dataset_type != "tabular":
        return X
    std = np.std(X, axis=0, keepdims=True)
    noise = np.random.normal(0, noise_ratio * std, size=X.shape)
    return X + noise


# =========================
# SCALING / PREPROCESSING AUGMENTATIONS
# =========================

_SCALERS = {
    'scale_standard': None,
    'scale_minmax': None,
    'scale_robust': None,
}


def _get_or_create_scaler(scaler_type, X_raw):
    if _SCALERS[scaler_type] is None:
        if scaler_type == 'scale_standard':
            scaler = StandardScaler()
        elif scaler_type == 'scale_minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'scale_robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        scaler.fit(X_raw)
        _SCALERS[scaler_type] = scaler
    return _SCALERS[scaler_type]


def scale_standard(X, dataset_type=None):
    if dataset_type != "tabular":
        return X
    scaler = _get_or_create_scaler('scale_standard', X)
    return scaler.transform(X)


def scale_minmax(X, dataset_type=None):
    if dataset_type != "tabular":
        return X
    scaler = _get_or_create_scaler('scale_minmax', X)
    return scaler.transform(X)


def scale_robust(X, dataset_type=None):
    if dataset_type != "tabular":
        return X
    scaler = _get_or_create_scaler('scale_robust', X)
    return scaler.transform(X)


# =========================
# IMAGE-SPECIFIC AUGMENTATION
# =========================

def geometric_transform(X, dataset_type=None):
    if dataset_type != "image":
        return X

    X_reshaped = X.reshape(-1, 28, 28)
    shifts = np.random.randint(-2, 3, size=(len(X_reshaped), 2))

    for i in range(len(X_reshaped)):
        X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][0], axis=0)
        X_reshaped[i] = np.roll(X_reshaped[i], shifts[i][1], axis=1)

    return X_reshaped.reshape(-1, 28 * 28)