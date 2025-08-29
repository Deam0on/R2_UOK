"""
Plotter script for visualizing outputs from analysis results in the output directory.
Generates plots for SHAP importances, PDPs, regression coefficients, ANOVA tables, and correlation matrices.
"""

import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_plotter_logger():
    """Set up logging for the plotter script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('plotter.log')
        ]
    )
    return logging.getLogger(__name__)


def find_latest_output_dir(base_output_dir="output"):
    """
    Find the latest output directory.
    
    Args:
        base_output_dir (str): Base output directory
        
    Returns:
        str: Path to the latest output directory
        
    Raises:
        FileNotFoundError: If no output directories found
    """
    if not os.path.exists(base_output_dir):
        raise FileNotFoundError(f"Base output directory not found: {base_output_dir}")
        
    subfolders = [f for f in os.listdir(base_output_dir) 
                  if f.startswith("results_") and os.path.isdir(os.path.join(base_output_dir, f))]
    if not subfolders:
        raise FileNotFoundError("No results subfolders found in output/.")
    
    # Sort and pick the latest
    latest_subfolder = sorted(subfolders)[-1]
    output_dir = os.path.join(base_output_dir, latest_subfolder)
    return output_dir


def plot_shap_importances(output_dir, top_n=20, logger=None):
    """
    Plot SHAP feature importances.
    
    Args:
        output_dir (str): Output directory path
        top_n (int): Number of top features to plot
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    shap_imp = os.path.join(output_dir, "shap_importance.csv")
    if os.path.exists(shap_imp):
        try:
            df = pd.read_csv(shap_imp)
            df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 5))
            plt.bar(df["feature"], df["mean_abs_shap"])
            plt.xticks(rotation=90)
            plt.title(f"Top {top_n} SHAP Feature Importances")
            plt.ylabel("Mean Absolute SHAP Value")
            plt.tight_layout()
            plt.show()
            
            logger.info(f"Plotted SHAP importances for top {len(df)} features")
        except Exception as e:
            logger.exception(f"Failed to plot SHAP importances: {e}")
    else:
        logger.warning("SHAP importance file not found")


def plot_1d_pdps(output_dir, logger=None):
    """
    Plot 1D Partial Dependence Plots.
    
    Args:
        output_dir (str): Output directory path
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    pdp_files = glob.glob(os.path.join(output_dir, "pdp_*.csv"))
    plotted_count = 0
    
    for pdp_file in pdp_files:
        # Skip 2D PDPs (they have underscores in the middle)
        base_name = os.path.basename(pdp_file)[4:-4]  # Remove 'pdp_' and '.csv'
        if "_" in base_name:
            continue
            
        try:
            df = pd.read_csv(pdp_file)
            if df.empty:
                logger.warning(f"Empty PDP file: {pdp_file}")
                continue
                
            feature = df.columns[0]
            plt.figure(figsize=(14, 7))
            plt.plot(df[feature], df["partial_dependence"])
            plt.xlabel(feature)
            plt.ylabel("Partial Dependence")
            plt.title(f"PDP for {feature}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            plotted_count += 1
            logger.debug(f"Plotted 1D PDP for {feature}")
            
        except Exception as e:
            logger.exception(f"Failed to plot 1D PDP from {pdp_file}: {e}")
    
    logger.info(f"Plotted {plotted_count} 1D PDPs")


def plot_2d_pdps(output_dir, logger=None):
    """
    Plot 2D Partial Dependence Plots.
    
    Args:
        output_dir (str): Output directory path
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    pdp2d_files = glob.glob(os.path.join(output_dir, "pdp_*_*.csv"))
    plotted_count = 0
    
    for pdp2d_file in pdp2d_files:
        try:
            df = pd.read_csv(pdp2d_file, index_col=0)
            if df.empty:
                logger.warning(f"Empty 2D PDP file: {pdp2d_file}")
                continue
                
            plt.figure(figsize=(14, 7))
            sns.heatmap(df, cmap="viridis", cbar_kws={'label': 'Partial Dependence'})
            plt.title(f"2D PDP: {df.index.name} vs {df.columns.name or df.columns[0]}")
            plt.xlabel(df.columns.name or df.columns[0])
            plt.ylabel(df.index.name)
            plt.tight_layout()
            plt.show()
            
            plotted_count += 1
            logger.debug(f"Plotted 2D PDP from {pdp2d_file}")
            
        except Exception as e:
            logger.exception(f"Failed to plot 2D PDP from {pdp2d_file}: {e}")
    
    logger.info(f"Plotted {plotted_count} 2D PDPs")


def plot_regression_coefficients(output_dir, top_n=20, logger=None):
    """
    Plot linear regression coefficients.
    
    Args:
        output_dir (str): Output directory path
        top_n (int): Number of top coefficients to plot
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    reg_files = glob.glob(os.path.join(output_dir, "linear_regression_*.csv"))
    plotted_count = 0
    
    for reg_file in reg_files:
        try:
            df = pd.read_csv(reg_file, index_col=0)
            if "Intercept" in df.index:
                df = df.drop("Intercept")
            
            if df.empty:
                logger.warning(f"No coefficients to plot in {reg_file}")
                continue
                
            df = df.sort_values("Coef.", key=abs, ascending=False).head(top_n)
            
            plt.figure(figsize=(14, 7))
            colors = ['red' if x < 0 else 'blue' for x in df["Coef."]]
            df["Coef."].plot(kind="bar", color=colors)
            plt.title(f"Top {top_n} Regression Coefficients: {os.path.basename(reg_file)}")
            plt.ylabel("Coefficient")
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            plotted_count += 1
            logger.debug(f"Plotted regression coefficients from {reg_file}")
            
        except Exception as e:
            logger.exception(f"Failed to plot regression coefficients from {reg_file}: {e}")
    
    logger.info(f"Plotted {plotted_count} regression coefficient plots")


def plot_anova_results(output_dir, top_n=20, logger=None):
    """
    Plot ANOVA results.
    
    Args:
        output_dir (str): Output directory path
        top_n (int): Number of top terms to plot
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    anova_files = glob.glob(os.path.join(output_dir, "anova_*_depth*.csv"))
    plotted_count = 0
    
    for anova_file in anova_files:
        try:
            df = pd.read_csv(anova_file, index_col=0)
            
            # Only plot significant terms
            if "P>|t|" in df.columns:
                sig = df[df["P>|t|"] < 0.05]
                if sig.empty:
                    logger.info(f"No significant terms in {anova_file}")
                    continue
                    
                sig = sig.sort_values("Coef.", key=abs, ascending=False).head(top_n)
                
                plt.figure(figsize=(14, 7))
                colors = ['red' if x < 0 else 'blue' for x in sig["Coef."]]
                sig["Coef."].plot(kind="bar", color=colors)
                plt.title(f"Top {top_n} ANOVA Significant Coefficients: {os.path.basename(anova_file)}")
                plt.ylabel("Coefficient")
                plt.xticks(rotation=45, ha='right')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                plotted_count += 1
                logger.debug(f"Plotted ANOVA results from {anova_file}")
            else:
                logger.warning(f"No P>|t| column found in {anova_file}")
                
        except Exception as e:
            logger.exception(f"Failed to plot ANOVA results from {anova_file}: {e}")
    
    logger.info(f"Plotted {plotted_count} ANOVA result plots")


def plot_correlation_matrix(output_dir, logger=None):
    """
    Plot correlation matrix.
    
    Args:
        output_dir (str): Output directory path
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    corr_file = os.path.join(output_dir, "correlation_matrix.csv")
    if os.path.exists(corr_file):
        try:
            df = pd.read_csv(corr_file, index_col=0)
            if df.empty:
                logger.warning("Empty correlation matrix file")
                return
                
            plt.figure(figsize=(10, 8))
            sns.heatmap(df, annot=True, cmap="coolwarm", center=0, 
                       cbar_kws={'label': 'Correlation'})
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.show()
            
            logger.info("Plotted correlation matrix")
            
        except Exception as e:
            logger.exception(f"Failed to plot correlation matrix: {e}")
    else:
        logger.warning("Correlation matrix file not found")


def main():
    """Main function to run all plotting functions."""
    logger = setup_plotter_logger()
    
    try:
        # Find latest output directory
        output_dir = find_latest_output_dir()
        logger.info(f"Using output directory: {output_dir}")
        print(f"Using output directory: {output_dir}")
        
        # Number of top features to plot (change as needed)
        TOP_N = 20
        
        # Generate all plots
        logger.info("Starting plot generation")
        
        plot_shap_importances(output_dir, TOP_N, logger)
        plot_1d_pdps(output_dir, logger)
        plot_2d_pdps(output_dir, logger)
        plot_regression_coefficients(output_dir, TOP_N, logger)
        plot_anova_results(output_dir, TOP_N, logger)
        plot_correlation_matrix(output_dir, logger)
        
        logger.info("All plots generated successfully")
        print("All plots generated successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Output directory not found: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in plotter: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
