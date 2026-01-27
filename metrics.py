import numpy as np
from scipy.stats import pearsonr

def compute_stability(attributions_list):
    attributions = np.array(attributions_list)

    # Feature-wise variance across runs
    feature_variance = np.var(attributions, axis=0)

    # Pairwise correlations across runs
    correlations = []
    for i in range(len(attributions)):
        for j in range(i + 1, len(attributions)):
            if np.std(attributions[i]) == 0 or np.std(attributions[j]) == 0:
                continue
            corr, _ = pearsonr(attributions[i], attributions[j])
            correlations.append(corr)

    return {
        "mean_feature_variance": float(np.mean(feature_variance)),
        "median_feature_variance": float(np.median(feature_variance)),
        "mean_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "min_correlation": float(np.min(correlations)) if correlations else 1.0,
    }
