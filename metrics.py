import numpy as np
from scipy.stats import pearsonr
import warnings


# =========================
# EXPLANATION STABILITY METRIC
# =========================

def compute_stability(attributions):
    if len(attributions) < 2:
        return {
            "mean_feature_variance": np.nan,
            "median_feature_variance": np.nan,
            "mean_correlation": np.nan,
            "min_correlation": np.nan,
        }

    # Stack: (n_runs, n_samples, n_features)
    A = np.stack(attributions, axis=0)
    if A.ndim != 3:
        raise ValueError(f"Expected shape (n_runs, n_samples, n_features), got {A.shape}")

    n_runs, n_samples, n_features = A.shape

    # Feature-wise variance across runs, then average over samples
    feature_variances = np.var(A, axis=0).mean(axis=0)

    mean_feature_variance = float(np.mean(feature_variances))
    median_feature_variance = float(np.median(feature_variances))

    # Pairwise Pearson correlations (flattened explanations)
    correlations = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            v1 = A[i].ravel()
            v2 = A[j].ravel()

            if np.std(v1) == 0 or np.std(v2) == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = pearsonr(v1, v2)

            if not np.isnan(corr):
                correlations.append(corr)

    if len(correlations) == 0:
        mean_corr = np.nan
        min_corr = np.nan
    else:
        mean_corr = float(np.mean(correlations))
        min_corr = float(np.min(correlations))

    return {
        "mean_feature_variance": mean_feature_variance,
        "median_feature_variance": median_feature_variance,
        "mean_correlation": mean_corr,
        "min_correlation": min_corr,
    }