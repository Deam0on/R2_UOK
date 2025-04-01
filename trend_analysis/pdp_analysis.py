from sklearn.inspection import PartialDependenceDisplay
from itertools import combinations
from math import ceil
import matplotlib.pyplot as plt

def plot_pdp(rf, X, feature_names, top_idx, significant_only=True):
    pdp_targets = [(i,) for i in top_idx] + list(combinations(top_idx, 2)) if significant_only else                   [(i,) for i in range(X.shape[1])] + list(combinations(range(X.shape[1]), 2))

    total_plots = len(pdp_targets)
    cols = 3
    rows_per_fig = 2
    plots_per_fig = cols * rows_per_fig
    n_figs = ceil(total_plots / plots_per_fig)

    for fig_num in range(n_figs):
        fig, axes = plt.subplots(rows_per_fig, cols, figsize=(18, 10))
        axes = axes.flatten()
        start, end = fig_num * plots_per_fig, min((fig_num + 1) * plots_per_fig, total_plots)

        for i, pdp_target in enumerate(pdp_targets[start:end]):
            try:
                PartialDependenceDisplay.from_estimator(rf, X, [pdp_target],
                                                        feature_names=feature_names,
                                                        kind="average", ax=axes[i])
            except Exception as e:
                axes[i].set_visible(False)

        for ax in axes[end-start:]:
            ax.set_visible(False)

        plt.suptitle(f"PDP Page {fig_num + 1}")
        plt.tight_layout()
        plt.show()
