import numpy as np
from scipy.stats import pearsonr
import warnings


def compute_stability(attributions):
    """
    Compute explanation stability across multiple training runs.

    Parameters
    ----------
    attributions : list of np.ndarray
        Each element has shape (n_samples, n_features)
        One explanation matrix per training run.

    Returns
    -------
    dict with:
        mean_feature_variance : float
        median_feature_variance : float
        mean_correlation : float
        min_correlation : float
    """
    # --------------------------------------------------
    # Basic checks
    # --------------------------------------------------
    if len(attributions) < 2:
        return {
            "mean_feature_variance": np.nan,
            "median_feature_variance": np.nan,
            "mean_correlation": np.nan,
            "min_correlation": np.nan,
        }

    # Shape: (n_runs, n_samples, n_features)
    A = np.stack(attributions, axis=0)
    if A.ndim != 3:
        raise ValueError(
            f"Expected attributions with shape (n_runs, n_samples, n_features), "
            f"but got shape {A.shape}"
        )

    n_runs, n_samples, n_features = A.shape

    # --------------------------------------------------
    # 1. Feature-wise variance across runs
    # --------------------------------------------------
    # Variance across runs → average over samples → per feature
    feature_variances = np.var(A, axis=0).mean(axis=0)

    mean_feature_variance = float(np.mean(feature_variances))
    median_feature_variance = float(np.median(feature_variances))

    # --------------------------------------------------
    # 2. Pairwise correlations between runs
    # --------------------------------------------------
    correlations = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            v1 = A[i].ravel()
            v2 = A[j].ravel()

            # Skip degenerate explanations
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
