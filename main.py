import numpy as np
from utils import load_tabular_dataset, load_image_dataset, save_results
from augmentations import no_augmentation, add_random_noise, scale_standard, scale_minmax, scale_robust, feature_jitter, geometric_transform
from models import train_model, evaluate_model
from explanations import compute_shap_explanations, compute_lime_explanations
from metrics import compute_stability



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
            add_random_noise
        ],
    },
}

# =========================
# NUM RUNS CONFIGURATION
# =========================

NUM_RUNS = {
    "iris": {
        "shap": 20,
        "lime": 20,
    },
    "wine": {
        "shap": 20,
        "lime": 20,
    },
    "mnist": {
        "shap": 20,
        "lime": 20,
    },
}

# =========================
# SINGLE EXPERIMENT
# =========================

def run_experiment(dataset_name, augmentation_fn, explainer, num_runs=5):

    # ────────────────────────────────────────────────
    # LOAD DATA
    # ────────────────────────────────────────────────
    if dataset_name in ["iris", "wine"]:
        X_train_base, X_test, y_train_base, y_test = load_tabular_dataset(dataset_name)
        dataset_type = "tabular"
    elif dataset_name == "mnist":
        X_train_base, X_test, y_train_base, y_test = load_image_dataset()
        dataset_type = "image"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ────────────────────────────────────────────────
    # RESET SCALERS FOR DATASET
    # ────────────────────────────────────────────────
    from augmentations import _SCALERS
    _SCALERS['scale_standard'] = None
    _SCALERS['scale_minmax'] = None
    _SCALERS['scale_robust'] = None
    print(f"[Reset] Cleared scalers for dataset '{dataset_name}'")

    # ────────────────────────────────────────────────
    # INITIALIZE SCALERS AND PREPARE SCALED TEST SETS
    # ────────────────────────────────────────────────
    X_test_standard = X_test.copy()
    X_test_minmax   = X_test.copy()
    X_test_robust   = X_test.copy()

    if dataset_type == "tabular":
        # Fit scalers on training data
        _ = scale_standard(X_train_base, dataset_type="tabular")
        _ = scale_minmax(X_train_base, dataset_type="tabular")
        _ = scale_robust(X_train_base, dataset_type="tabular")

        # Apply transformations to test data
        X_test_standard = scale_standard(X_test, dataset_type="tabular")
        X_test_minmax = scale_minmax(X_test, dataset_type="tabular")
        X_test_robust = scale_robust(X_test, dataset_type="tabular")

        # Debug print
        print(f"[{dataset_name}] scale_standard test mean/std: "
              f"{X_test_standard.mean(axis=0)[:2]} / {X_test_standard.std(axis=0)[:2]}")

    attributions = []
    accuracies = []

    expl_subsample_size = len(X_test)   # will be overridden below if needed
    lime_num_features = None
    lime_num_samples = None

    if dataset_name == "mnist":
        expl_subsample_size = 300
        if explainer == "lime":
            lime_num_features = 200
            lime_num_samples  = 5000

    # ────────────────────────────────────────────────
    # MULTIPLE RUNS
    # ────────────────────────────────────────────────
    for run in range(num_runs):
        np.random.seed(run)

        X_train_aug = augmentation_fn(X_train_base, dataset_type=dataset_type)

        # Select the right test set
        if augmentation_fn is scale_standard:
            X_test_current = X_test_standard
        elif augmentation_fn is scale_minmax:
            X_test_current = X_test_minmax
        elif augmentation_fn is scale_robust:
            X_test_current = X_test_robust
        else:
            X_test_current = X_test.copy()

        model = train_model(
            X_train_aug,
            y_train_base,
            dataset_type=dataset_type
        )

        acc = evaluate_model(model, X_test_current, y_test)
        accuracies.append(acc)

        X_test_expl = X_test_current[:expl_subsample_size]

        # explanations...
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

    print("Experiments complete. Results saved.")