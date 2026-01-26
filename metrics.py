import numpy as np
from scipy.stats import pearsonr

def compute_stability(attributions_list):
    """
    Stability metrics:
    - Mean variance: sensitivity of feature importance across runs
    - Mean Pearson correlation: agreement between runs
    """
    attributions = np.array(attributions_list)
    mean_attrib = np.mean(attributions, axis=0)
    variance = np.var(attributions, axis=0)

    correlations = []
    for i in range(len(attributions)):
        for j in range(i + 1, len(attributions)):
            if np.std(attributions[i]) == 0 or np.std(attributions[j]) == 0:
                continue
            corr, _ = pearsonr(attributions[i], attributions[j])
            correlations.append(corr)

    return {
        "mean_variance": float(np.mean(variance)),
        "mean_correlation": float(np.mean(correlations)) if correlations else 1.0
    }
