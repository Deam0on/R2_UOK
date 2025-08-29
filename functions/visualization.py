"""
Visualization utilities for correlation matrices and PCA plots.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def show_correlation(df, input_numerics, save_plots=False, output_file="correlation_matrix.png"):
    """
    Plot a correlation matrix for the given numeric input columns.

    Parameters:
    - df: The DataFrame containing the data.
    - input_numerics: List of numeric input column names to include in the correlation matrix.
    - save_plots: Boolean indicating whether to save the plot as a file.
    - output_file: Name of the output file if save_plots is True.
    
    Raises:
    - ValueError: If input parameters are invalid.
    """
    logger = logging.getLogger(__name__)
    
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
            
        if not input_numerics:
            logger.warning("No numeric input columns provided for correlation matrix")
            return
            
        # Check if all columns exist in DataFrame
        missing_cols = [col for col in input_numerics if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns not found in DataFrame: {missing_cols}")
            input_numerics = [col for col in input_numerics if col in df.columns]
            
        if not input_numerics:
            logger.warning("No valid numeric columns found for correlation matrix")
            return
        
        logger.debug(f"Creating correlation matrix for columns: {input_numerics}")
        
        # Calculate correlation matrix
        corr = df[input_numerics].corr()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr, 
            annot=True, 
            cmap="coolwarm", 
            center=0,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title("Correlation Matrix (Numeric Inputs)")
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved correlation matrix to: {output_file}")
            except Exception as e:
                logger.exception(f"Failed to save correlation matrix: {e}")
                raise
        else:
            plt.show()
            
        # Save correlation matrix as CSV for further analysis
        if save_plots:
            try:
                csv_file = output_file.replace('.png', '.csv')
                corr.to_csv(csv_file)
                logger.debug(f"Saved correlation matrix data to: {csv_file}")
            except Exception as e:
                logger.exception(f"Failed to save correlation matrix CSV: {e}")
        
        logger.info(f"Successfully created correlation matrix for {len(input_numerics)} variables")
        
    except Exception as e:
        logger.exception(f"Failed to create correlation matrix: {e}")
        raise


def plot_pca(X, pca, save_plots=False, output_file="pca_variance.png", feature_names=None, output_dir=None):
    """
    Plot cumulative explained variance for PCA components and save component loadings.

    Parameters:
    - X: The input data to perform PCA on.
    - pca: The PCA object (from sklearn.decomposition) to use for the transformation.
    - save_plots: Boolean indicating whether to save the plot as a file.
    - output_file: Name of the output file if save_plots is True.
    - feature_names: List of feature names for component loadings analysis.
    - output_dir: Output directory for saving component loadings CSV files.
    
    Returns:
    - dict: Dictionary containing PCA analysis results including component loadings
    
    Raises:
    - ValueError: If input parameters are invalid.
    """
    logger = logging.getLogger(__name__)
    
    try:
        if X is None:
            raise ValueError("Input data X is None")
            
        if not hasattr(pca, 'fit_transform'):
            raise ValueError("PCA object must have fit_transform method")
        
        logger.debug(f"Performing PCA on data with shape: {X.shape}")
        
        # Fit PCA and transform data
        X_pca = pca.fit_transform(X)
        
        # Calculate cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_variance_ratio)
        
        logger.debug(f"PCA results: {len(explained_variance_ratio)} components, "
                    f"total variance explained: {cumsum_var[-1]:.3f}")
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot individual component variance
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Individual Component Variance")
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, marker="o", linewidth=2, markersize=6)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance (PCA)")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Variance')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Variance')
        plt.legend()
        
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved PCA plot to: {output_file}")
            except Exception as e:
                logger.exception(f"Failed to save PCA plot: {e}")
                raise
        else:
            plt.show()
        
        # Log some useful information
        n_components_80 = np.argmax(cumsum_var >= 0.8) + 1
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        logger.info(f"PCA analysis: {n_components_80} components explain 80% variance, "
                   f"{n_components_95} components explain 95% variance")
        
        # Analyze and save component loadings
        pca_results = {
            'n_components_80': n_components_80,
            'n_components_95': n_components_95,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumsum_var
        }
        
        if feature_names is not None and output_dir is not None:
            try:
                # Get component loadings (eigenvectors)
                components = pca.components_
                
                # Create component loadings DataFrame
                loadings_df = pd.DataFrame(
                    components.T,
                    columns=[f'PC{i+1}' for i in range(components.shape[0])],
                    index=feature_names
                )
                
                # Save all component loadings
                loadings_file = os.path.join(output_dir, "pca_component_loadings.csv")
                loadings_df.to_csv(loadings_file)
                logger.info(f"Saved PCA component loadings to: {loadings_file}")
                
                # Create summary of top contributors for key components
                summary_data = []
                
                # Analyze the components that explain 80% and 95% variance
                key_components = min(n_components_95, 15)  # Look at up to 15 components max
                
                for i in range(key_components):
                    pc_name = f'PC{i+1}'
                    loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
                    top_features = loadings.head(5)  # Top 5 contributors
                    
                    summary_data.append({
                        'Component': pc_name,
                        'Variance_Explained': f"{explained_variance_ratio[i]:.3f}",
                        'Cumulative_Variance': f"{cumsum_var[i]:.3f}",
                        'Top_5_Features': ', '.join([f"{feat}({val:.3f})" for feat, val in top_features.items()])
                    })
                
                # Save summary
                summary_df = pd.DataFrame(summary_data)
                summary_file = os.path.join(output_dir, "pca_components_summary.csv")
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved PCA components summary to: {summary_file}")
                
                # Print key findings to console
                print(f"\nPCA Analysis Results:")
                print(f"- {n_components_80} components explain 80% of variance")
                print(f"- {n_components_95} components explain 95% of variance")
                print(f"\nTop contributors to first {min(5, key_components)} components:")
                
                for i in range(min(5, key_components)):
                    pc_name = f'PC{i+1}'
                    loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
                    top_3 = loadings.head(3)
                    print(f"  {pc_name} ({explained_variance_ratio[i]:.1%} variance): {', '.join([f'{feat}({val:.2f})' for feat, val in top_3.items()])}")
                
                pca_results.update({
                    'loadings_df': loadings_df,
                    'summary_df': summary_df
                })
                
            except Exception as e:
                logger.exception(f"Failed to analyze PCA component loadings: {e}")
        
        return pca_results
        
    except Exception as e:
        logger.exception(f"Failed to create PCA plot: {e}")
        raise
