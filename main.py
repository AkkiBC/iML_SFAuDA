import numpy as np
from utils import load_tabular_dataset, load_image_dataset, save_results
from augmentations import no_augmentation, add_random_noise, scale_standard, scale_minmax, scale_robust, feature_jitter, bootstrap_resample, geometric_transform
from models import train_model, evaluate_model
from explanations import compute_shap_explanations, compute_lime_explanations
from metrics import compute_stability
from plots import plot_accuracy_vs_worst_case_stability

def run_experiment(dataset_name, aug_strategy, expl_method, num_runs=5):
    if dataset_name in ['iris', 'wine']:
        X_train_base, X_test, y_train_base, y_test = load_tabular_dataset(dataset_name)
        dataset_type = 'tabular'
    else:  # 'mnist'
        X_train_base, X_test, y_train_base, y_test = load_image_dataset()
        dataset_type = 'image'
    
    attributions = []
    accuracies = []
    
    for run in range(num_runs):
        np.random.seed(run)  # Vary seed for variability
        X_train_aug = aug_strategy(X_train_base, dataset_type=dataset_type)
        model = train_model(X_train_aug, y_train_base, dataset_type=dataset_type)
        
        acc = evaluate_model(model, X_test, y_test)
        accuracies.append(acc)
        
        if expl_method == 'shap':
            attrib = compute_shap_explanations(model, X_train_aug, X_test)
        elif expl_method == 'lime':
            attrib = compute_lime_explanations(model, X_train_aug, X_test)
        attributions.append(attrib)
    
    stability = compute_stability(attributions)
    stability['dataset'] = dataset_name
    stability['augmentation'] = aug_strategy.__name__
    stability['expl_method'] = expl_method
    stability['mean_accuracy'] = np.mean(accuracies)
    
    return stability

if __name__ == "__main__":
    configs = [
        # Iris - SHAP
        ("iris", no_augmentation, "shap"),
        ("iris", add_random_noise, "shap"),
        ("iris", scale_standard, "shap"),
        ("iris", scale_minmax, "shap"),
        ("iris", scale_robust, "shap"),
        ("iris", feature_jitter, "shap"),
        ("iris", bootstrap_resample, "shap"),
        ('iris', add_random_noise, 'shap'),
        # ('iris', geometric_transform, 'shap'),
        # Iris LIME
        ("iris", no_augmentation, "lime"),
        ("iris", scale_standard, "lime"),
        ("iris", scale_minmax, "lime"),
        # Wine 
        ("wine", no_augmentation, "shap"),
        ("wine", scale_standard, "shap"),
        ("wine", feature_jitter, "shap"),
        ("wine", bootstrap_resample, "shap"),
        ("wine", no_augmentation, "lime"),
        # Add more: MNIST, geometric_transform, etc.
    ]
    results = []
    for config in configs:
        result = run_experiment(*config)
        results.append(result)
    save_results(results)
    plot_accuracy_vs_worst_case_stability()
    print("Experiments complete. Results saved to results/stability_scores.csv")