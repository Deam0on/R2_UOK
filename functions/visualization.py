import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_correlation(df, input_numerics, save_plots=False):
    """
    Plot a correlation matrix for the given numeric input columns.
    If save_plots is True, saves the plot as 'correlation_matrix.png'.

    Parameters:
    - df: The DataFrame containing the data.
    - input_numerics: List of numeric input column names to include in the correlation matrix.
    - save_plots: Boolean indicating whether to save the plot as a file.
    """
    if input_numerics:
        corr = df[input_numerics].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix (Numeric Inputs)")
        plt.tight_layout()
        if save_plots:
            plt.savefig("correlation_matrix.png")
    # else:
    #     plt.show()


def plot_pca(X, pca, save_plots=False):
    """
    Plot cumulative explained variance for PCA components.
    If save_plots is True, saves the plot as 'pca_variance.png'.

    Parameters:
    - X: The input data to perform PCA on.
    - pca: The PCA object (from sklearn.decomposition) to use for the transformation.
    - save_plots: Boolean indicating whether to save the plot as a file.
    """
    X_pca = pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig("pca_variance.png")
    # else:
    #     plt.show()
