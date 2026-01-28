import shap
import lime
import lime.lime_tabular
import numpy as np
from tqdm import tqdm


def compute_shap_explanations(model, X_train, X_test, dataset_name, background_size=20):

    # =========================
    # BACKGROUND SELECTION
    # =========================
    if dataset_name == "mnist":
        background = shap.kmeans(X_train, background_size)
    elif dataset_name == "wine":
        background = shap.kmeans(X_train, background_size)
    else:
        background = X_train  # keep full for small tabular like iris

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)

    # =========================
    # MULTI-CLASS HANDLING (UNCHANGED)
    # =========================
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = np.mean(np.stack(shap_values, axis=-1), axis=-1)

    if shap_values.ndim == 3:
        shap_values = np.mean(shap_values, axis=-1)

    # RETURN: (n_samples, n_features)
    return shap_values


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

    for x in tqdm(X_test):
        exp = explainer.explain_instance(
            x,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        # Take explanation for the predicted class (or top class)
        pred_class = exp.available_labels()[0]  # usually the predicted one
        weights = np.zeros(X_train.shape[1])

        # robust against MNIST multi-class
        label = exp.available_labels()[0]

        for feature_idx, weight in exp.as_map()[label]:
            weights[feature_idx] = weight

        explanations.append(weights)

    # RETURN: (n_samples, n_features)
    return np.array(explanations)
