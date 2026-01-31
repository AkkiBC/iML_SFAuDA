import shap
import lime
import lime.lime_tabular
import numpy as np
from tqdm import tqdm


def compute_shap_explanations(model, X_train, X_test, dataset_name, background_size=20):
    if dataset_name == "mnist":
        background = shap.kmeans(X_train, background_size)
    elif dataset_name == "wine":
        background = shap.kmeans(X_train, background_size)
    else:
        background = X_train

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)

    # ─── Binary-safe handling ───────────────────────────────────────
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            # Binary classification → usually take positive class
            shap_values = shap_values[1]
        else:
            # Unexpected → fallback to average (rare)
            shap_values = np.mean(np.stack(shap_values, axis=-1), axis=-1)
    # ────────────────────────────────────────────────────────────────

    if shap_values.ndim == 3:
        shap_values = np.mean(shap_values, axis=-1)   # shouldn't happen in binary

    return shap_values   # expected shape: (n_samples, n_features)


def compute_lime_explanations(model, X_train, X_test, num_features=None, num_samples=None):
    if num_features is None:
        num_features = X_train.shape[1]
    if num_samples is None:
        num_samples = 3000

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        discretize_continuous=False
    )

    explanations = []

    for x in tqdm(X_test, desc="LIME explanations"):
        exp = explainer.explain_instance(
            x,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        # ─── Binary-safe: take the explanation for the predicted class ───
        if len(exp.available_labels()) == 0:
            weights = np.zeros(X_train.shape[1])  # fallback
        else:
            label = exp.available_labels()[0]     # highest prob class
            weights = np.zeros(X_train.shape[1])
            for f_idx, w in exp.as_map()[label]:
                weights[f_idx] = w
        # ────────────────────────────────────────────────────────────────

        explanations.append(weights)

    return np.array(explanations)
