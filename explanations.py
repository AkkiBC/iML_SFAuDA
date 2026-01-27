import shap
import lime
import lime.lime_tabular
import numpy as np

def compute_shap_explanations(model, X_train, X_test, dataset_name, background_size=20):
    if dataset_name == "wine":
        background = shap.kmeans(X_train, background_size)
    else:
        background = X_train

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values

def compute_lime_explanations(model, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        discretize_continuous=False
    )

    explanations = []

    for x in X_test:
        exp = explainer.explain_instance(
            x,
            model.predict_proba,
            num_features=X_train.shape[1]
        )

        weights = np.zeros(X_train.shape[1])
        for feature_idx, weight in exp.as_map()[1]:
            weights[feature_idx] = weight

        explanations.append(weights)

    return np.array(explanations)

