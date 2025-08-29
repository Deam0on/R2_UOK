import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_correlation(df, input_numerics, save_plots=False):
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
    X_pca = pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig("pca_variance.png")
    # else:
    #     plt.show()
