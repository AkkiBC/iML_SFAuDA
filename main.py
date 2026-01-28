import numpy as np
from utils import load_tabular_dataset, load_image_dataset, make_performance_table, make_stability_table, save_results
from augmentations import no_augmentation, add_random_noise, scale_standard, scale_minmax, scale_robust, feature_jitter, bootstrap_resample, geometric_transform
from models import train_model, evaluate_model
from explanations import compute_shap_explanations, compute_lime_explanations
from metrics import compute_stability
from plots import plot_accuracy_vs_worst_case_stability, plot_accuracy_vs_mean_stability

def run_experiment(dataset_name, aug_strategy, expl_method, num_runs=5):
    if dataset_name in ['iris', 'wine']:
        X_train_base, X_test, y_train_base, y_test = load_tabular_dataset(dataset_name)
        dataset_type = 'tabular'
    else:  # 'mnist'
        X_train_base, X_test, y_train_base, y_test = load_image_dataset()
        dataset_type = 'image'
    
    attributions = []
    accuracies = []

    # ── Configuration: per-dataset + per-method tuning ──
    subsample_size = len(X_test)  # default: full test set
    lime_num_features = None      # default: all features
    lime_num_samples = 3000       # default for tabular; reduce for MNIST

    if dataset_name == 'mnist':
        subsample_size = 100      # explain only 100 test images → huge speedup
        if expl_method == 'lime':
            lime_num_features = 50    # limit to top 50 pixels (critical for 784-dim)
            lime_num_samples = 1500   # reduce perturbations (default 5000 is too slow)

    # SHAP already handles background summarization in compute_shap_explanations

    for run in range(num_runs):
        np.random.seed(run)  # Vary seed for variability
        X_train_aug = aug_strategy(X_train_base, dataset_type=dataset_type)
        model = train_model(X_train_aug, y_train_base, dataset_type=dataset_type)
        
        # Subsample test set only for explanations (accuracy still on full set)
        X_test_sub = X_test[:subsample_size]

        acc = evaluate_model(model, X_test, y_test)  # full test set for fair accuracy
        accuracies.append(acc)

        print(f"Run {run+1}/{num_runs} | Dataset: {dataset_name} | Aug: {aug_strategy.__name__} | Expl: {expl_method} | Test samples for expl: {len(X_test_sub)}")

        if expl_method == 'shap':
            attrib = compute_shap_explanations(model, X_train_aug, X_test_sub, dataset_name)
        elif expl_method == 'lime':
            attrib = compute_lime_explanations(
                model,
                X_train_aug,
                X_test_sub,
                num_features=lime_num_features,
                num_samples=lime_num_samples
            )
        else:
            raise ValueError(f"Unknown expl_method: {expl_method}")

        attributions.append(attrib)
    
    stability = compute_stability(attributions)
    stability['dataset'] = dataset_name
    stability['augmentation'] = aug_strategy.__name__
    stability['expl_method'] = expl_method
    stability['mean_accuracy'] = np.mean(accuracies)
    
    return stability

if __name__ == "__main__":
    configs = [
    # Tabular – SHAP
    ("iris", no_augmentation, "shap", 8),     # increase to 8–10 runs
    ("iris", add_random_noise, "shap", 8),
    ("iris", scale_standard, "shap", 8),      # if you kept scaling augs
    ("iris", feature_jitter, "shap", 8),
    ("wine", no_augmentation, "shap", 8),
    ("wine", add_random_noise, "shap", 8),

    # Tabular – LIME (fewer runs because slower)
    ("iris", no_augmentation, "lime", 5),
    ("iris", add_random_noise, "lime", 5),
    ("wine", no_augmentation, "lime", 5),

    # Image (MNIST) – start with SHAP only, fewer runs
    ("mnist", no_augmentation, "shap", 3),
    ("mnist", geometric_transform, "shap", 3),
    ("mnist", add_random_noise, "shap", 3),   # add later if time
    ("mnist", no_augmentation, "lime", 3),
    ("mnist", geometric_transform, "lime", 3),
]
    results = []
    for config in configs:
        result = run_experiment(*config)
        results.append(result)
    save_results(results)
    plot_accuracy_vs_worst_case_stability()
    plot_accuracy_vs_mean_stability()
    make_performance_table()
    make_stability_table()
    print("Experiments complete. Results saved to results/stability_scores.csv")