
import pandas as pd
from sklearn.inspection import partial_dependence
from itertools import combinations
from math import ceil

def plot_pdp(rf, X, feature_names, top_idx, significant_only=True, save_plots=False):
    import os
    os.makedirs('output', exist_ok=True)
    pdp_targets = [(i,) for i in top_idx] if significant_only else [(i,) for i in range(X.shape[1])]
    # 1D PDPs
    for pdp_target in pdp_targets:
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            values = pd_result['values'][0]
            averages = pd_result['average'][0]
            feature = feature_names[pdp_target[0]]
            df_pdp = pd.DataFrame({feature: values, 'partial_dependence': averages})
            df_pdp.to_csv(f'output/pdp_{feature}.csv', index=False)
        except Exception:
            continue

    # 2D PDPs (pairwise)
    pdp_targets_2d = list(combinations(top_idx, 2)) if significant_only else list(combinations(range(X.shape[1]), 2))
    for pdp_target in pdp_targets_2d:
        try:
            pd_result = partial_dependence(rf, X, [pdp_target], kind="average")
            x_values, y_values = pd_result['values']
            z_values = pd_result['average'][0]  # shape: (len(x_values), len(y_values))
            feature1 = feature_names[pdp_target[0]]
            feature2 = feature_names[pdp_target[1]]
            df_pdp2d = pd.DataFrame(z_values, index=x_values, columns=y_values)
            df_pdp2d.index.name = feature1
            df_pdp2d.columns.name = feature2
            df_pdp2d.to_csv(f'output/pdp_{feature1}_{feature2}.csv')
        except Exception:
            continue
