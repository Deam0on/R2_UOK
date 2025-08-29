"""
SHAP analysis utilities for model interpretability.
"""
import os
import numpy as np
import pandas as pd
import shap

def explain_shap(
    rf, X, feature_names, significant_only=True, save_plots=False, check_additivity=True, output_dir="output"
):
    """
    Explain the SHAP values of a fitted model.

    Parameters:
    - rf: The fitted model (e.g., a scikit-learn random forest model).
    - X: The input data for prediction, as a Pandas DataFrame or NumPy array.
    - feature_names: The names of the features, as a list of strings.
    - significant_only: If True, only the top features with the highest mean absolute SHAP values are considered.
    - save_plots: If True, save the SHAP values and feature importances as CSV files.
    - check_additivity: If True, check the additivity of the SHAP values.

    Returns:
    - mean_shap: The mean absolute SHAP values for each feature.
    - top_idx: The indices of the top features with the highest mean absolute SHAP values.
    - shap_values: The SHAP values for each feature and each instance in the input data.
    """
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-5:][::-1]
        # Save SHAP values and mean SHAP importances as CSV
        pd.DataFrame(shap_values, columns=feature_names).to_csv(
            os.path.join(output_dir, "shap_values.csv"), index=False
        )
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_shap}).to_csv(
            os.path.join(output_dir, "shap_importance.csv"), index=False
        )
        return mean_shap, top_idx, shap_values
    except Exception as e:
        print(f"Error in explain_shap: {e}")
        return None, None, None
