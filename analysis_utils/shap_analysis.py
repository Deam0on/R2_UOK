"""
SHAP analysis utilities for model interpretability.
"""

import logging
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
    - output_dir: Directory where to save output files. Default is "output".

    Returns:
    - mean_shap: The mean absolute SHAP values for each feature.
    - top_idx: The indices of the top features with the highest mean absolute SHAP values.
    - shap_values: The SHAP values for each feature and each instance in the input data.
    
    Raises:
    - ValueError: If model or data is invalid.
    - OSError: If output directory cannot be created.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not hasattr(rf, 'predict'):
            raise ValueError("Model must have a 'predict' method")
            
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match X columns ({X.shape[1]})")
        
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created/verified output directory: {output_dir}")
        except OSError as e:
            logger.exception(f"Failed to create output directory: {output_dir}")
            raise OSError(f"Cannot create output directory: {output_dir}") from e
        
        logger.info("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(rf)
        logger.debug("Successfully created TreeExplainer")
        
        logger.info("Computing SHAP values...")
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)
        logger.debug(f"SHAP values shape: {shap_values.shape}")
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-5:][::-1]
        logger.info(f"Top 5 features by SHAP: {[feature_names[i] for i in top_idx]}")
        
        # Save SHAP values and mean SHAP importances as CSV
        if save_plots:
            try:
                shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_values_path = os.path.join(output_dir, "shap_values.csv")
                shap_values_df.to_csv(shap_values_path, index=False)
                logger.debug(f"Saved SHAP values to: {shap_values_path}")
                
                importance_df = pd.DataFrame({
                    "feature": feature_names, 
                    "mean_abs_shap": mean_shap
                })
                importance_path = os.path.join(output_dir, "shap_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                logger.debug(f"Saved SHAP importances to: {importance_path}")
                
            except Exception as e:
                logger.exception("Failed to save SHAP results to CSV")
                raise
        
        logger.info("SHAP analysis completed successfully")
        return mean_shap, top_idx, shap_values
        
    except Exception as e:
        logger.exception(f"Error in explain_shap: {e}")
        print(f"Error in explain_shap: {e}")
        return None, None, None
