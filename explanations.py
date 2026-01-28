import shap
import lime
import lime.lime_tabular
import numpy as np
from tqdm import tqdm


<<<<<<< HEAD
def compute_shap_explanations(model, X_train, X_test, dataset_name, background_size=20):

    # =========================
    # BACKGROUND SELECTION
    # =========================
    if dataset_name == "mnist":
        background = shap.kmeans(X_train, background_size)
    elif dataset_name == "wine":
        background = shap.kmeans(X_train, background_size)
=======
def compute_shap_explanations(model, X_train, X_test, dataset_name, background_size=50):
    if dataset_name in ["mnist"]:
        # For high-dim image data: summarize background aggressively
        background = shap.kmeans(X_train, background_size)  # or shap.sample(X_train, background_size)
    elif dataset_name == "wine":
        background = shap.kmeans(X_train, background_size)  # optional, wine has only 13 features
>>>>>>> bbc79a5575862d98d9e91da17425d85b50b8715d
    else:
        background = X_train  # keep full for small tabular like iris

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)

<<<<<<< HEAD
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

=======
    # For multi-class, usually take class 1 or mean abs; adjust based on your needs
    if isinstance(shap_values, list):
        shap_values = np.array([np.abs(sv) for sv in shap_values]).mean(axis=0)  # mean abs over classes
    else:
        shap_values = np.abs(shap_values)

    # Return mean attribution per feature (for stability comparison)
    return np.mean(shap_values, axis=0)

def compute_lime_explanations(model, X_train, X_test, num_features=None, num_samples=3000):
    if num_features is None:
        num_features = X_train.shape[1]

>>>>>>> bbc79a5575862d98d9e91da17425d85b50b8715d
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        discretize_continuous=False
    )

    explanations = []

<<<<<<< HEAD
    for x in tqdm(X_test):
=======
    for i, x in enumerate(X_test):
        print(f"  Explaining instance {i+1}/{len(X_test)} ...")
>>>>>>> bbc79a5575862d98d9e91da17425d85b50b8715d
        exp = explainer.explain_instance(
            x,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        # Take explanation for the predicted class (or top class)
        pred_class = exp.available_labels()[0]  # usually the predicted one
        weights = np.zeros(X_train.shape[1])
<<<<<<< HEAD

        # robust against MNIST multi-class
        label = exp.available_labels()[0]

        for feature_idx, weight in exp.as_map()[label]:
            weights[feature_idx] = weight

        explanations.append(weights)

    # RETURN: (n_samples, n_features)
    return np.array(explanations)
=======
        for feat_idx, weight in exp.as_map()[pred_class]:
            weights[feat_idx] = weight

        explanations.append(weights)

    return np.mean(np.abs(explanations), axis=0)  # mean abs attribution per feature

>>>>>>> bbc79a5575862d98d9e91da17425d85b50b8715d
