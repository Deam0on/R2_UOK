"""
Partial Dependence Plot (PDP) analysis utilities.
"""

import logging
import os
from itertools import combinations
from math import ceil

import pandas as pd
from sklearn.inspection import partial_dependence


def plot_pdp(rf, X, feature_names, top_idx, significant_only=True, save_plots=False, output_dir="output"):
    """
    Generate and save Partial Dependence Plots (PDPs) for a given model and dataset.

    Parameters:
    - rf: The trained model (e.g., a RandomForestRegressor or RandomForestClassifier).
    - X: The input data used for training the model, as a pandas DataFrame or numpy array.
    - feature_names: The names of the features in the input data.
    - top_idx: The indices of the top features for which to plot PDPs.
    - significant_only: If True, plot PDPs only for the significant features. Default is True.
    - save_plots: If True, save the plots as CSV files in the "output" directory. Default is False.
    - output_dir: Directory where to save the output files. Default is "output".

    Returns:
    - None
    
    Raises:
    - OSError: If output directory cannot be created.
    - ValueError: If input parameters are invalid.
    """
    logger = logging.getLogger(__name__)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created/verified output directory: {output_dir}")
    except OSError as e:
        logger.exception(f"Failed to create output directory: {output_dir}")
        raise OSError(f"Cannot create output directory: {output_dir}") from e
    
    if not hasattr(rf, 'predict'):
        raise ValueError("Model must have a 'predict' method")
        
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match X columns ({X.shape[1]})")
    
    if top_idx is None or len(top_idx) == 0:
        logger.warning("No top indices provided, using all features")
        top_idx = list(range(X.shape[1]))

    pdp_targets = (
        [(i,) for i in top_idx]
        if significant_only
        else [(i,) for i in range(X.shape[1])]
    )
    
    logger.info(f"Generating 1D PDPs for {len(pdp_targets)} features")
    
    # 1D PDPs
    for pdp_target in pdp_targets:
        feature = feature_names[pdp_target[0]]
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            logger.debug(f"PDP result keys: {list(pd_result.keys())}")
            
            # Handle different scikit-learn versions
            if "values" in pd_result:
                # Older scikit-learn versions
                values = pd_result["values"][0]
                averages = pd_result["average"][0]
            elif "grid_values" in pd_result:
                # Newer scikit-learn versions
                values = pd_result["grid_values"][0]
                averages = pd_result["average"][0]
            else:
                # Try to extract from available keys
                logger.warning(f"Unknown PDP result structure for {feature}. Available keys: {list(pd_result.keys())}")
                # Skip this feature if we can't understand the structure
                continue
                
            df_pdp = pd.DataFrame({feature: values, "partial_dependence": averages})
            
            output_file = os.path.join(output_dir, f"pdp_{feature}.csv")
            df_pdp.to_csv(output_file, index=False)
            logger.debug(f"Saved 1D PDP for {feature} to {output_file}")
            
        except Exception as e:
            logger.exception(f"Error generating PDP for {feature}: {e}")
            if 'pd_result' in locals():
                logger.debug(f"pd_result keys: {list(pd_result.keys())}")
            else:
                logger.debug("pd_result not defined")
            continue

    # 2D PDPs (pairwise)
    pdp_targets_2d = (
        list(combinations(top_idx, 2))
        if significant_only
        else list(combinations(range(X.shape[1]), 2))
    )
    
    logger.info(f"Generating 2D PDPs for {len(pdp_targets_2d)} feature pairs")
    
    for pdp_target in pdp_targets_2d:
        feature1 = feature_names[pdp_target[0]]
        feature2 = feature_names[pdp_target[1]]
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            logger.debug(f"2D PDP result keys: {list(pd_result.keys())}")
            
            # Handle different scikit-learn versions
            if "values" in pd_result:
                # Older scikit-learn versions
                x_values, y_values = pd_result["values"]
                z_values = pd_result["average"][0]  # shape: (len(x_values), len(y_values))
            elif "grid_values" in pd_result:
                # Newer scikit-learn versions
                x_values, y_values = pd_result["grid_values"]
                z_values = pd_result["average"][0]  # shape: (len(x_values), len(y_values))
            else:
                # Try to extract from available keys
                logger.warning(f"Unknown 2D PDP result structure for {feature1} and {feature2}. Available keys: {list(pd_result.keys())}")
                # Skip this feature pair if we can't understand the structure
                continue
                
            df_pdp2d = pd.DataFrame(z_values, index=x_values, columns=y_values)
            df_pdp2d.index.name = feature1
            df_pdp2d.columns.name = feature2
            
            output_file = os.path.join(output_dir, f"pdp_{feature1}_{feature2}.csv")
            df_pdp2d.to_csv(output_file)
            logger.debug(f"Saved 2D PDP for {feature1} and {feature2} to {output_file}")
            
        except Exception as e:
            logger.exception(f"Error generating 2D PDP for {feature1} and {feature2}: {e}")
            if 'pd_result' in locals():
                logger.debug(f"pd_result keys: {list(pd_result.keys())}")
            else:
                logger.debug("pd_result not defined")
            continue
    
    logger.info("PDP generation completed")
