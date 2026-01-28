import numpy as np
from utils import load_tabular_dataset, load_image_dataset, make_performance_table, make_stability_table, save_results
from augmentations import no_augmentation, add_random_noise, scale_standard, scale_minmax, scale_robust, feature_jitter, geometric_transform
from models import train_model, evaluate_model
from explanations import compute_shap_explanations, compute_lime_explanations
from metrics import compute_stability
from plots import plot_all_stability_intervals


# =========================
# EXPERIMENT SETUP
# =========================

DATASETS = ["iris", "wine", "mnist"]

EXPLAINERS = ["shap", "lime"]

# =========================
# AUGMENTATION CONFIGURATION
# =========================

AUGMENTATIONS_CONFIG = {
    "iris": {
        "shap": [
            no_augmentation,
            add_random_noise,
            feature_jitter,
            scale_standard,
            scale_minmax,
            scale_robust,
        ],
        "lime": [
            no_augmentation,
            add_random_noise,
            feature_jitter,
            scale_standard,
            scale_minmax,
            scale_robust,
        ],
    },
    "wine": {
        "shap": [
            no_augmentation,
            add_random_noise,
            feature_jitter,
            scale_standard,
            scale_minmax,
            scale_robust,
        ],
        "lime": [
            no_augmentation,
            add_random_noise,
            feature_jitter,
            scale_standard,
            scale_minmax,
            scale_robust,
        ],
    },
    "mnist": {
        "shap": [
            no_augmentation,
            geometric_transform,
            add_random_noise,
        ],
        "lime": [
            no_augmentation,
            geometric_transform,
        ],
    },
}

# =========================
# NUM RUNS CONFIGURATION
# =========================

NUM_RUNS = {
    "iris": {
        "shap": 8,
        "lime": 5,
    },
    "wine": {
        "shap": 8,
        "lime": 5,
    },
    "mnist": {
        "shap": 3,
        "lime": 3,
    },
}

# =========================
# SINGLE EXPERIMENT
# =========================

def run_experiment(dataset_name, augmentation_fn, explainer, num_runs=5):

    # =========================
    # LOAD DATA
    # =========================
    if dataset_name in ["iris", "wine"]:
        X_train_base, X_test, y_train_base, y_test = load_tabular_dataset(dataset_name)
        dataset_type = "tabular"
    elif dataset_name == "mnist":  # ← MNIST
        X_train_base, X_test, y_train_base, y_test = load_image_dataset()
        dataset_type = "image"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    attributions = []
    accuracies = []

    # =========================
    # MNIST-SPECIFIC CONFIG
    # =========================
    expl_subsample_size = len(X_test)
    lime_num_features = None
    lime_num_samples = None

    if dataset_name == "mnist":  # ← MNIST
        expl_subsample_size = 100
        if explainer == "lime":
            lime_num_features = 50
            lime_num_samples = 1500

    # =========================
    # RUNS
    # =========================
    for run in range(num_runs):
        np.random.seed(run)

        X_train_aug = augmentation_fn(X_train_base, dataset_type=dataset_type)

        model = train_model(
            X_train_aug,
            y_train_base,
            dataset_type=dataset_type
        )

        acc = evaluate_model(model, X_test, y_test)
        accuracies.append(acc)

        X_test_expl = X_test[:expl_subsample_size]  # ← MNIST (noop for tabular)

        if explainer == "shap":
            attrib = compute_shap_explanations(
                model=model,
                X_train=X_train_aug,
                X_test=X_test_expl,
                dataset_name=dataset_name
            )

        elif explainer == "lime":
            attrib = compute_lime_explanations(
                model=model,
                X_train=X_train_aug,
                X_test=X_test_expl,
                num_features=lime_num_features,
                num_samples=lime_num_samples
            )

        else:
            raise ValueError("Unknown explainer")

        if attrib.ndim != 2:
            raise ValueError(
                f"Explainer {explainer} returned invalid shape {attrib.shape}"
            )

        attributions.append(attrib)

    # =========================
    # STABILITY
    # =========================
    stability = compute_stability(attributions)

    stability.update({
        "dataset": dataset_name,
        "augmentation": augmentation_fn.__name__,
        "expl_method": explainer,
        "mean_accuracy": float(np.mean(accuracies))
    })

    return stability

if __name__ == "__main__":

    results = []

    for dataset in DATASETS:
        for explainer in EXPLAINERS:
            for augmentation in AUGMENTATIONS_CONFIG[dataset][explainer]:

                # Skip incompatible augs automatically (optional safety)
                if dataset == "mnist" and augmentation.__name__ == "scale_robust":
                    continue

                print(f"Running: {dataset} | {explainer} | {augmentation.__name__}")

                num_runs = NUM_RUNS[dataset][explainer]

                result = run_experiment(
                    dataset_name=dataset,
                    augmentation_fn=augmentation,
                    explainer=explainer,
                    num_runs=num_runs,
                )

                results.append(result)

    save_results(results)

    plot_all_stability_intervals()
    make_performance_table()
    make_stability_table()

    print("Experiments complete. Results saved.")
