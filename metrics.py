import numpy as np
from scipy.stats import pearsonr

def compute_stability(attributions_list):
    # attributions_list: list of (num_features,) arrays
    mean_attrib = np.mean(attributions_list, axis=0)
    variance = np.var(attributions_list, axis=0)
    correlations = []
    for i in range(len(attributions_list)):
        for j in range(i+1, len(attributions_list)):
            corr, _ = pearsonr(attributions_list[i], attributions_list[j])
            correlations.append(corr)
    return {
        'mean_variance': np.mean(variance),
        'mean_correlation': np.mean(correlations) if correlations else 1.0
    }