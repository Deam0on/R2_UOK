import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_shap(rf, X, feature_names, significant_only=True):
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-5:][::-1]

    if significant_only:
        shap.summary_plot(shap_values[:, top_idx], X[:, top_idx],
                          feature_names=[feature_names[i] for i in top_idx])
    else:
        shap.summary_plot(shap_values, X, feature_names=feature_names)

    return mean_shap, top_idx, shap_values
