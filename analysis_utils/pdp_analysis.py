"""
Partial Dependence Plot (PDP) analysis utilities.
"""

from itertools import combinations
from math import ceil
import os
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

    Returns:
    - None
    """
 

    os.makedirs(output_dir, exist_ok=True)
    pdp_targets = (
        [(i,) for i in top_idx]
        if significant_only
        else [(i,) for i in range(X.shape[1])]
    )
    # 1D PDPs
    for pdp_target in pdp_targets:
        feature = feature_names[pdp_target[0]]
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            values = pd_result["values"][0]
            averages = pd_result["average"][0]
            df_pdp = pd.DataFrame({feature: values, "partial_dependence": averages})
            df_pdp.to_csv(os.path.join(output_dir, f"pdp_{feature}.csv"), index=False)
        except Exception as e:
            print(f"Error generating PDP for {feature}: {e}")
            print(f"pd_result keys: {list(pd_result.keys()) if 'pd_result' in locals() else 'pd_result not defined'}")
            continue

    # 2D PDPs (pairwise)
    pdp_targets_2d = (
        list(combinations(top_idx, 2))
        if significant_only
        else list(combinations(range(X.shape[1]), 2))
    )
    for pdp_target in pdp_targets_2d:
        feature1 = feature_names[pdp_target[0]]
        feature2 = feature_names[pdp_target[1]]
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            x_values, y_values = pd_result["values"]
            z_values = pd_result["average"][0]  # shape: (len(x_values), len(y_values))
            df_pdp2d = pd.DataFrame(z_values, index=x_values, columns=y_values)
            df_pdp2d.index.name = feature1
            df_pdp2d.columns.name = feature2
            df_pdp2d.to_csv(os.path.join(output_dir, f"pdp_{feature1}_{feature2}.csv"))
        except Exception as e:
            print(f"Error generating 2D PDP for {feature1} and {feature2}: {e}")
            print(f"pd_result keys: {list(pd_result.keys()) if 'pd_result' in locals() else 'pd_result not defined'}")
            continue
