"""
Summary script for aggregating and visualizing parameter-level importances from SHAP, ANOVA, and PDP outputs.
"""

import glob
import logging
import os
import re
from collections import defaultdict
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.inspection import permutation_importance

# Suppress sklearn feature name warnings for cleaner output
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestRegressor was fitted without feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.simplefilter("ignore", UserWarning)


def setup_summary_logger(output_dir=None):
    """Set up logging for the summary script."""
    if output_dir and os.path.exists(output_dir):
        log_file = os.path.join(output_dir, 'Logs', 'summary.log')
    else:
        log_file = 'summary.log'
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


def save_and_show_plot(output_dir, plot_name, logger, show_plot=True):
    """
    Save plot to Visualizations folder and optionally show it.
    
    Args:
        output_dir (str): Output directory path
        plot_name (str): Name of the plot file (without extension)
        logger: Logger instance
        show_plot (bool): Whether to display the plot
    """
    try:
        # Ensure Visualizations directory exists
        viz_dir = os.path.join(output_dir, "Visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save the plot
        plot_path = os.path.join(viz_dir, f"{plot_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot to {plot_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close to save memory if not showing
            
    except Exception as e:
        logger.warning(f"Failed to save plot {plot_name}: {e}")
        if show_plot:
            plt.show()  # Still show even if save failed


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {config_path}") from e


def find_latest_results_dir(base_output_dir="output"):
    """
    Find the latest results directory.
    
    Args:
        base_output_dir (str): Base output directory
        
    Returns:
        str: Path to the latest results directory
        
    Raises:
        FileNotFoundError: If no results directories found
    """
    if not os.path.exists(base_output_dir):
        raise FileNotFoundError(f"Base output directory not found: {base_output_dir}")
        
    subfolders = [f for f in os.listdir(base_output_dir) if f.startswith("results_")]
    if not subfolders:
        raise FileNotFoundError("No results subfolders found in output/.")
    
    # Sort and pick the latest
    latest_subfolder = sorted(subfolders)[-1]
    output_dir = os.path.join(base_output_dir, latest_subfolder)
    return output_dir


def get_parameter_from_feature(feature_name, param_names):
    """
    Extract parameter name from feature name.
    
    Args:
        feature_name (str): Name of the feature
        param_names (list): List of parameter names
        
    Returns:
        str or None: Parameter name if found, None otherwise
    """
    for p in param_names:
        if p in feature_name:
            return p
    return None


def process_shap_importances(output_dir, param_names, logger):
    """
    Process SHAP importance files and generate parameter-level summaries.
    
    Args:
        output_dir (str): Output directory path
        param_names (list): List of parameter names
        logger: Logger instance
    """
    shap_imp_path = os.path.join(output_dir, "SHAP", "shap_importance.csv")
    if os.path.exists(shap_imp_path):
        try:
            df = pd.read_csv(shap_imp_path)
            df_copy = df.copy()  # Create explicit copy to avoid SettingWithCopyWarning
            df_copy["parameter"] = df_copy["feature"].apply(
                lambda x: get_parameter_from_feature(x, param_names)
            )
            grouped = (
                df_copy.groupby("parameter")["mean_abs_shap"]
                .sum()
                .reindex(param_names)
                .dropna()
                .sort_values(ascending=False)
            )
            print("\nSHAP Parameter Importances (sum of mean_abs_shap):\n", grouped)
            
            try:
                plt.figure(figsize=(8, 4))
                grouped.head(20).plot(kind="bar")
                plt.title("Top Parameters by SHAP Importance")
                plt.ylabel("Summed mean_abs_shap")
                plt.tight_layout()
                save_and_show_plot(output_dir, "shap_parameter_importance", logger)
            except Exception as e:
                logger.exception(f"Failed to plot SHAP importances: {e}")
        except Exception as e:
            logger.exception(f"Failed to process SHAP importances: {e}")
    else:
        print("No SHAP importances found.")
        logger.warning("No SHAP importances found.")


def process_anova_files(output_dir, param_names, logger):
    """
    Process ANOVA result files and generate parameter-level summaries.
    
    Args:
        output_dir (str): Output directory path
        param_names (list): List of parameter names
        logger: Logger instance
    """
    anova_files = glob.glob(os.path.join(output_dir, "ANOVA", "anova_*_depth*.csv"))
    if anova_files:
        for anova_file in anova_files:
            try:
                df = pd.read_csv(anova_file, index_col=0)
                # Only use significant terms
                if "P>|t|" in df.columns:
                    sig = df[df["P>|t|"] < 0.05].copy()  # Create explicit copy

                    sig["parameter"] = sig.index.map(
                        lambda x: get_parameter_from_feature(x, param_names)
                    )
                    grouped = (
                        sig.groupby("parameter")["Coef."]
                        .apply(lambda x: x.abs().sum())
                        .reindex(param_names)
                        .dropna()
                        .sort_values(ascending=False)
                    )
                    filename = os.path.basename(anova_file)
                    match = re.search(r"depth(\d+)", filename)
                    depth_str = f" (depth {match.group(1)})" if match else ""
                    print(
                        f"\nANOVA Parameter Importances for {filename}:\n",
                        grouped,
                    )
                    if not grouped.empty:
                        try:
                            plt.figure(figsize=(8, 4))
                            grouped.head(20).plot(kind="bar")
                            plt.title(
                                f"Top Parameters by ANOVA (|sum Coef|): {filename}{depth_str}"
                            )
                            plt.ylabel("Sum |Coef| (significant)")
                            plt.tight_layout()
                            # Generate unique plot name based on filename and depth
                            plot_name = f"anova_parameters_{filename.replace(' ', '_')}{depth_str.replace(' ', '_')}"
                            save_and_show_plot(output_dir, plot_name, logger)
                        except Exception as e:
                            logger.exception(f"Failed to plot ANOVA results for {filename}: {e}")
                    else:
                        print(f"No significant terms to plot for {filename}{depth_str}.")
            except Exception as e:
                logger.exception(f"Failed to process ANOVA file {anova_file}: {e}")
    else:
        print("No ANOVA files found.")
        logger.warning("No ANOVA files found.")


def process_regression_files(output_dir, param_names, logger):
    """
    Process linear regression result files and generate parameter-level summaries.
    
    Args:
        output_dir (str): Output directory path
        param_names (list): List of parameter names
        logger: Logger instance
    """
    reg_files = glob.glob(os.path.join(output_dir, "Models", "linear_regression_*.csv"))
    if reg_files:
        for reg_file in reg_files:
            try:
                df = pd.read_csv(reg_file, index_col=0)
                if "Intercept" in df.index:
                    df = df.drop("Intercept")

                df_copy = df.copy()  # Create explicit copy
                df_copy["parameter"] = df_copy.index.map(
                    lambda x: get_parameter_from_feature(x, param_names)
                )
                grouped = (
                    df_copy.groupby("parameter")["Coef."]
                    .apply(lambda x: x.abs().sum())
                    .reindex(param_names)
                    .dropna()
                    .sort_values(ascending=False)
                )
                print(
                    f"\nRegression Parameter Importances for {os.path.basename(reg_file)}:\n",
                    grouped,
                )
                if not grouped.empty:
                    try:
                        plt.figure(figsize=(8, 4))
                        grouped.head(20).plot(kind="bar")
                        plt.title(
                            f"Top Parameters by Regression (|sum Coef|): {os.path.basename(reg_file)}"
                        )
                        plt.ylabel("Sum |Coef|")
                        plt.tight_layout()
                        # Generate unique plot name based on filename
                        plot_name = f"regression_parameters_{os.path.basename(reg_file).replace('.csv', '').replace(' ', '_')}"
                        save_and_show_plot(output_dir, plot_name, logger)
                    except Exception as e:
                        logger.exception(f"Failed to plot regression results for {reg_file}: {e}")
                else:
                    print(
                        f"No regression coefficients to plot for {os.path.basename(reg_file)}."
                    )
            except Exception as e:
                logger.exception(f"Failed to process regression file {reg_file}: {e}")
    else:
        print("No regression coefficient files found.")
        logger.warning("No regression coefficient files found.")


def process_permutation_importance(output_dir, logger):
    """
    Process permutation importance if model and data are available.
    
    Args:
        output_dir (str): Output directory path
        logger: Logger instance
    """
    try:
        # Look for any available model files
        models_dir = os.path.join(output_dir, "Models")
        model_files = glob.glob(os.path.join(models_dir, "rf_model_*.joblib"))
        
        if not model_files:
            print("Permutation importance: no Random Forest models found.")
            logger.info("Permutation importance: no Random Forest models found.")
            return
            
        # Use the first model found
        model_path = model_files[0]
        target_name = os.path.basename(model_path).replace("rf_model_", "").replace(".joblib", "")
        
        X_path = os.path.join(models_dir, f"X_{target_name}.csv")
        y_path = os.path.join(models_dir, f"y_{target_name}.csv")
        
        if os.path.exists(model_path) and os.path.exists(X_path) and os.path.exists(y_path):
            try:
                model = joblib.load(model_path)
                X = pd.read_csv(X_path)
                y = pd.read_csv(y_path).squeeze()
                result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
                importances = pd.Series(result.importances_mean, index=X.columns)
                # Group by parameter
                importances_grouped = importances.groupby(
                    importances.index.map(
                        lambda x: x.split("__")[1] if "__" in x else x.split("_")[0]
                    )
                ).sum()
                print(
                    f"\nPermutation Importances for {target_name} (grouped):\n",
                    importances_grouped.sort_values(ascending=False),
                )
                try:
                    plt.figure(figsize=(8, 4))
                    importances_grouped.sort_values(ascending=False).head(20).plot(kind="bar")
                    plt.title(f"Top Parameters by Permutation Importance ({target_name})")
                    plt.ylabel("Mean Importance")
                    plt.tight_layout()
                    # Generate unique plot name based on target
                    plot_name = f"permutation_importance_{target_name.replace(' ', '_')}"
                    save_and_show_plot(output_dir, plot_name, logger)
                except Exception as e:
                    logger.exception(f"Failed to plot permutation importance: {e}")
            except Exception as e:
                logger.exception(f"Failed to compute permutation importance: {e}")
        else:
            print(f"Permutation importance: required files not found for {target_name}.")
            logger.info(f"Permutation importance: required files not found for {target_name}.")
    except ImportError as e:
        print("joblib or sklearn not installed; skipping permutation importance.")
        logger.warning(f"Failed to import required packages for permutation importance: {e}")
    except Exception as e:
        logger.exception(f"Error in permutation importance processing: {e}")


def process_pdp_files(output_dir, param_names, logger, top_n=10):
    """
    Process PDP result files and generate condensed summaries.
    
    Args:
        output_dir (str): Output directory path
        param_names (list): List of parameter names
        logger: Logger instance
        top_n (int): Number of top features to show in detail
    """
    pdp_dir = os.path.join(output_dir, "PDP")
    if not os.path.exists(pdp_dir):
        print("No PDP directory found.")
        logger.warning("No PDP directory found.")
        return

    # Get 1D PDP files (avoid 2D files which have underscores in feature names)
    pdp_files_1d = []
    for file in os.listdir(pdp_dir):
        if file.startswith("pdp_") and file.endswith(".csv"):
            # Count underscores to differentiate 1D from 2D PDPs
            feature_part = file[4:-4]  # Remove 'pdp_' and '.csv'
            if feature_part.count('_') <= 3:  # 1D PDPs have fewer underscores
                pdp_files_1d.append(os.path.join(pdp_dir, file))
    
    if not pdp_files_1d:
        print("No 1D PDP files found.")
        logger.warning("No 1D PDP files found.")
        return

    print(f"\nPDP Analysis Summary ({len(pdp_files_1d)} 1D PDP files found)")
    logger.info(f"Processing {len(pdp_files_1d)} 1D PDP files")

    # Strategy 1: PDP Effect Magnitude Summary
    try:
        pdp_effects = {}
        pdp_ranges = {}
        
        for pdp_file in pdp_files_1d:
            try:
                df = pd.read_csv(pdp_file)
                if len(df.columns) >= 2:
                    feature_name = df.columns[0]
                    pdp_values = df.columns[1]
                    
                    # Calculate PDP effect magnitude (max - min)
                    pdp_effect = df[pdp_values].max() - df[pdp_values].min()
                    pdp_effects[feature_name] = pdp_effect
                    pdp_ranges[feature_name] = (df[pdp_values].min(), df[pdp_values].max())
                    
            except Exception as e:
                logger.warning(f"Failed to process PDP file {pdp_file}: {e}")

        if pdp_effects:
            # Create DataFrame for easier analysis
            pdp_summary = pd.DataFrame([
                {
                    'feature': feature,
                    'pdp_effect_magnitude': effect,
                    'pdp_min': pdp_ranges[feature][0],
                    'pdp_max': pdp_ranges[feature][1],
                    'parameter': get_parameter_from_feature(feature, param_names)
                }
                for feature, effect in pdp_effects.items()
            ]).sort_values('pdp_effect_magnitude', ascending=False)

            print(f"\nTop {top_n} Features by PDP Effect Magnitude:")
            print(pdp_summary.head(top_n)[['feature', 'pdp_effect_magnitude', 'parameter']].to_string(index=False))

            # Group by parameter
            if param_names:
                param_effects = pdp_summary.groupby('parameter')['pdp_effect_magnitude'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
                param_effects = param_effects.dropna()
                
                print(f"\nPDP Effects by Parameter (top {min(10, len(param_effects))}):")
                print(param_effects.head(10).to_string())

                # Plot parameter-level PDP effects
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot 1: Top individual features
                    top_features = pdp_summary.head(top_n)
                    ax1.barh(range(len(top_features)), top_features['pdp_effect_magnitude'])
                    ax1.set_yticks(range(len(top_features)))
                    ax1.set_yticklabels([f"{row['feature'][:30]}..." if len(row['feature']) > 30 else row['feature'] 
                                        for _, row in top_features.iterrows()], fontsize=8)
                    ax1.set_xlabel('PDP Effect Magnitude')
                    ax1.set_title(f'Top {top_n} Features by PDP Effect')
                    ax1.invert_yaxis()

                    # Plot 2: Parameter-level effects
                    param_effects_plot = param_effects.head(10)
                    ax2.barh(range(len(param_effects_plot)), param_effects_plot['mean'])
                    ax2.set_yticks(range(len(param_effects_plot)))
                    ax2.set_yticklabels(param_effects_plot.index, fontsize=8)
                    ax2.set_xlabel('Mean PDP Effect Magnitude')
                    ax2.set_title('Parameter-Level PDP Effects')
                    ax2.invert_yaxis()
                    
                    plt.tight_layout()
                    save_and_show_plot(output_dir, "pdp_effect_magnitude_summary", logger)
                    
                except Exception as e:
                    logger.exception(f"Failed to plot PDP summary: {e}")

            # Strategy 2: Detailed plots for top features
            try:
                print(f"\nGenerating detailed PDP plots for top {min(6, top_n)} features...")
                
                top_features_detailed = pdp_summary.head(min(6, top_n))
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes = axes.flatten()
                
                for i, (_, row) in enumerate(top_features_detailed.iterrows()):
                    if i >= 6:
                        break
                        
                    feature = row['feature']
                    # Find the corresponding PDP file
                    pdp_file = None
                    for f in pdp_files_1d:
                        if feature in f:
                            pdp_file = f
                            break
                    
                    if pdp_file:
                        try:
                            df = pd.read_csv(pdp_file)
                            feature_col = df.columns[0]
                            pdp_col = df.columns[1]
                            
                            axes[i].plot(df[feature_col], df[pdp_col], 'b-', linewidth=2, marker='o', markersize=3)
                            axes[i].set_xlabel(feature_col)
                            axes[i].set_ylabel('Partial Dependence')
                            axes[i].set_title(f'{feature}\nEffect: {row["pdp_effect_magnitude"]:.4f}', fontsize=10)
                            axes[i].grid(True, alpha=0.3)
                            
                        except Exception as e:
                            axes[i].text(0.5, 0.5, f'Error loading\n{feature}', 
                                       ha='center', va='center', transform=axes[i].transAxes)
                            logger.warning(f"Failed to plot detailed PDP for {feature}: {e}")
                
                # Hide empty subplots
                for i in range(len(top_features_detailed), 6):
                    axes[i].set_visible(False)
                
                plt.suptitle(f'Detailed PDP Plots - Top {min(6, len(top_features_detailed))} Features by Effect Magnitude', fontsize=14)
                plt.tight_layout()
                save_and_show_plot(output_dir, "pdp_detailed_top_features", logger)
                
            except Exception as e:
                logger.exception(f"Failed to create detailed PDP plots: {e}")

    except Exception as e:
        logger.exception(f"Failed to process PDP files: {e}")
        print(f"Error processing PDP files: {e}")


def main():
    """Main function to run the summary analysis."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate analysis summary with auto-saved plots")
    parser.add_argument("output_dir", nargs="?", default=None, 
                       help="Specific output directory to analyze (default: latest in output/)")
    args = parser.parse_args()
    
    # Initialize logger first (without output dir initially)
    logger = setup_summary_logger()
    
    try:
        # Determine output directory
        if args.output_dir:
            if os.path.exists(args.output_dir):
                output_dir = args.output_dir
                print(f"Using specified output directory: {output_dir}")
            else:
                print(f"Error: Specified directory {args.output_dir} does not exist")
                return
        else:
            # Find latest results directory
            base_output_dir = "output"
            output_dir = find_latest_results_dir(base_output_dir)
            print(f"Using latest output directory: {output_dir}")
        
        # Reinitialize logger with correct output directory
        logger = setup_summary_logger(output_dir)
        logger.info(f"Using output directory: {output_dir}")
        logger.info("Starting summary analysis with auto-saved plots")

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        try:
            config = load_config(config_path)
            param_names = config["input_categoricals"] + config["input_numerics"]
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            print(f"Warning: Failed to load config, using default parameter names")
            param_names = []

        # Process different types of results
        print("\nGenerating summary plots (auto-saved to Visualizations folder)...")
        process_shap_importances(output_dir, param_names, logger)
        process_anova_files(output_dir, param_names, logger)  
        process_regression_files(output_dir, param_names, logger)
        process_permutation_importance(output_dir, logger)
        process_pdp_files(output_dir, param_names, logger)

        print(f"\nSummary complete. All plots saved to: {os.path.join(output_dir, 'Visualizations')}")
        logger.info("Summary analysis completed successfully with all plots saved")
        
    except Exception as e:
        logger.exception(f"Failed to complete summary analysis: {e}")
        print(f"Error: Failed to complete summary analysis: {e}")


if __name__ == "__main__":
    main()
