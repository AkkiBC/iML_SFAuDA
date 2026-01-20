import numpy as np

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