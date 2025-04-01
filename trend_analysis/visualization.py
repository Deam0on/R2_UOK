import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_correlation(df, input_numerics):
    if input_numerics:
        sns.heatmap(df[input_numerics].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix (Numeric Inputs)")
        plt.tight_layout()
        plt.show()

def plot_pca(X, pca):
    X_pca = pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Components")
    plt.ylabel("Explained Variance")
    plt.grid()
    plt.tight_layout()
    plt.show()
