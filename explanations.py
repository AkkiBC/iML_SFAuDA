import shap
import lime
import lime.lime_tabular
import numpy as np

def compute_shap_explanations(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    return np.mean(np.abs(shap_values), axis=0)  # Average absolute attributions per feature

def compute_lime_explanations(model, X_train, X_test, num_features=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode="classification")
    attributions = []
    for x in X_test:
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num_features)
        attributions.append([v for _, v in sorted(exp.as_list())])
    return np.mean(np.abs(attributions), axis=0)  # Average absolute attributions