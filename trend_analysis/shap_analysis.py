import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def explain_shap(rf, X, feature_names, significant_only=True, save_plots=False, check_additivity=True):
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-5:][::-1]
    # Save SHAP values and mean SHAP importances as CSV
    pd.DataFrame(shap_values, columns=feature_names).to_csv('output/shap_values.csv', index=False)
    pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_shap}).to_csv('output/shap_importance.csv', index=False)
    return mean_shap, top_idx, shap_values
