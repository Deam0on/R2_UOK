"""
Main analysis pipeline for trend analysis, regression, feature importance, and statistical evaluation.
"""

import logging
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from analysis_utils.anova import run_anova
from analysis_utils.modeling import fit_linear_model, fit_random_forest
from analysis_utils.pdp_analysis import plot_pdp
from analysis_utils.preprocess import build_preprocessor, load_and_clean
from analysis_utils.shap_analysis import explain_shap
from analysis_utils.utils import (
    check_imbalance,
    clean_anova_terms,
    clean_linear_terms,
    generate_auto_summary,
    print_summary,
    print_table,
    setup_logger,
)
from functions.visualization import plot_pca, show_correlation


def group_rare_categories(df, cat_columns, threshold=10):
    """
    For each categorical column, group all but the top N most frequent categories into 'Other'.

    Args:
        df (pd.DataFrame): The input dataframe.
        cat_columns (list): List of categorical column names.
        threshold (int): The number of top categories to keep. All others will be grouped as 'Other'.

    Returns:
        pd.DataFrame: The dataframe with rare categories grouped.
        
    Raises:
        Exception: If any error occurs during category grouping.
    """
    logger = logging.getLogger()
    
    for col in cat_columns:
        try:
            # Convert to string for consistency
            df[col] = df[col].astype(str)
            value_counts = df[col].value_counts()
            to_keep = set(value_counts.nlargest(threshold).index)
            df[col] = df[col].apply(lambda x: x if x in to_keep else "Other")
            # Log the result
            n_unique = df[col].nunique()
            print(
                f"After grouping, '{col}' has {n_unique} unique values: {df[col].unique()}"
            )
            logger.debug(f"Successfully grouped categories for column '{col}'")
        except Exception as e:
            logger.exception(f"Failed to group categories for column '{col}': {e}")
            # Continue with other columns even if this one fails
    return df


def main(config=None):
    """
    Main function to execute the analysis pipeline.

    Args:
        config (dict, optional): Configuration dictionary. If None, the function will raise an error.
            Required keys:
            - input_categoricals: List of categorical input column names
            - input_numerics: List of numeric input column names  
            - output_targets: List of target output column names
            - output_dir: Directory for saving output files
            Optional keys:
            - csv_path: Path to CSV file (if not using SQL)
            - dropna_required: Boolean to drop NA values
            - use_wls: Boolean to use weighted least squares
            - significant_only: Boolean to show only significant results
            - reference_levels: Dict mapping categorical columns to reference levels
    """
    print("[DEBUG] Entered main()")
    
    # Get output directory from config
    output_dir = config.get('output_dir', 'output')
    
    # Enhanced logging setup: file and console
    try:
        setup_logger()
        logger = logging.getLogger()
        # Remove all handlers associated with the root logger object (avoid duplicate logs)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add file handler to output directory
        log_file = os.path.join(output_dir, "Logs", "analysis.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    except Exception as e:
        print(f"[ERROR] Failed to set up logging: {e}")
        return

    logger.info("Starting trend analysis main function.")
    
    try:
        if config is None:
            logger.error("No config provided to main(). Please use the CLI or pass a config dict.")
            raise ValueError(
                "No config provided to main(). Please use the CLI or pass a config dict."
            )
            
        # Get output directory from config
        output_dir = config.get('output_dir', 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        logger.debug(f"Using output directory: {output_dir}")
        logger.debug(f"Config keys: {list(config.keys())}")

        # Load and clean data
        try:
            df = load_and_clean(
                config.get("csv_path"),
                config["input_categoricals"] + config["input_numerics"],
                config["output_targets"],
                dropna_required=config.get("dropna_required", True),
                config=config,
            )
            logger.debug(f"Successfully loaded data with shape: {df.shape}")
        except Exception as e:
            logger.exception(f"Failed to load and clean data: {e}")
            raise
            
        # Group rare categories in high-cardinality categoricals
        try:
            df = group_rare_categories(df, config["input_categoricals"], threshold=10)
            logger.debug("Successfully grouped rare categories")
        except Exception as e:
            logger.exception(f"Failed to group rare categories: {e}")
            # Continue without grouping if this fails
            
        # Warn if cardinality is still high after grouping
        for col in config["input_categoricals"]:
            try:
                n_unique = df[col].nunique()
                if n_unique > 20:
                    logger.warning(
                        f"Categorical variable '{col}' still has high cardinality ({n_unique} unique values) after grouping."
                    )
            except Exception as e:
                logger.warning(f"Failed to check cardinality for column '{col}': {e}")
                
        # Ensure categorical columns are strings
        try:
            for col in config["input_categoricals"]:
                df[col] = df[col].astype(str)
            logger.debug("Successfully converted categorical columns to strings")
        except Exception as e:
            logger.exception(f"Failed to convert categorical columns to strings: {e}")
            raise

        logger.info(f"Loaded dataframe with shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"First 5 rows:\n{df.head()}\n")
        
        try:
            results_path = os.path.join(output_dir, "analysis_results.txt")
            # Save head of dataframe to results file and as CSV
            df.head().to_csv(os.path.join(output_dir, "data_head.csv"), index=False)
            logger.debug(f"Saved data head to: {os.path.join(output_dir, 'data_head.csv')}")
        except Exception as e:
            logger.exception(f"Failed to save data head: {e}")
            results_path = "analysis_results.txt"  # Fallback to current directory

        try:
            with open(results_path, "w", encoding="utf-8") as results_file:
                results_file.write("First 5 rows of data:\n")
                results_file.write(df.head().to_string())
                results_file.write("\n\n")
            logger.debug(f"Initialized results file: {results_path}")
        except Exception as e:
            logger.exception(f"Failed to initialize results file: {e}")

        if not config.get("dropna_required", False):
            try:
                logger.info("Imputing missing values for numeric columns.")
                imputer = SimpleImputer(strategy="mean")
                df[config["input_numerics"]] = imputer.fit_transform(
                    df[config["input_numerics"]]
                )
                logger.debug("Successfully imputed missing values")
            except Exception as e:
                logger.exception(f"Failed to impute missing values: {e}")
                # Continue without imputation if this fails

        logger.info("Processing categorical reference levels...")
        try:
            for col in config["input_categoricals"]:
                ref = str(config.get("reference_levels", {}).get(col))
                logger.debug(f"Reference for {col}: {ref}")
        except Exception as e:
            logger.exception(f"Failed to process reference levels: {e}")

        if config.get("run_imbalance_check", False):
            try:
                logger.info("Checking for feature imbalance...")
                skewed_cols = check_imbalance(df, config)
                logger.debug(f"Skewed columns: {skewed_cols}")
            except Exception as e:
                logger.exception(f"Failed imbalance check: {e}")

        logger.info("Building preprocessor and transforming features...")
        try:
            preprocessor = build_preprocessor(
                config["input_categoricals"], config["input_numerics"]
            )
            X = preprocessor.fit_transform(
                df[config["input_categoricals"] + config["input_numerics"]]
            )
            logger.debug("Successfully built and applied preprocessor")
        except Exception as e:
            logger.exception(f"Failed to build preprocessor or transform features: {e}")
            raise

        # If you have feature_names (from your preprocessor or pipeline), use them; otherwise, use generic names
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            logger.debug(f"Feature names: {feature_names[:5]}...")  # Log first 5 feature names
        except Exception as e:
            logger.exception(f"Failed to get feature names: {e}")
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(f"Transformed feature matrix X with shape: {X.shape}")
        logger.debug(f"First 5 rows of X:\n{X[:5]}\n")

        logger.info("Plotting correlation matrix...")
        try:
            output_file = os.path.join(output_dir, "Visualizations", "correlation_matrix.png")
            show_correlation(df, config["input_numerics"], save_plots=True, output_file=output_file)
        except Exception as e:
            logger.exception(f"Failed to plot correlation matrix: {e}")
            
        logger.info("Plotting PCA...")
        try:
            output_file = os.path.join(output_dir, "Visualizations", "pca_variance.png")
            pca_results = plot_pca(X, PCA(), save_plots=True, output_file=output_file, 
                                 feature_names=feature_names, output_dir=output_dir)
        except Exception as e:
            logger.exception(f"Failed to plot PCA: {e}")
            pca_results = None

        target_summaries = {}

        # Process each output target
        orig_stdout = sys.stdout
        try:
            with open(results_path, "a", encoding="utf-8") as results_file:
                sys.stdout = results_file
                
                for output in config["output_targets"]:
                    target_summary = None
                    try:
                        pd.DataFrame(X, columns=feature_names).to_csv(
                            os.path.join(output_dir, f"X_{output}.csv"), index=False
                        )
                        logger.debug(f"Saved feature matrix for {output}")
                    except Exception as e:
                        logger.exception(f"Failed to save feature matrix for {output}: {e}")
                        
                    try:
                        logger.info(f"Processing output target: {output}")
                        if output not in df.columns:
                            logger.error(
                                f"Output target '{output}' not found in dataframe columns. Skipping."
                            )
                            continue
                        y = df[output]
                        if y.isnull().all():
                            logger.error(
                                f"All values for output target '{output}' are NaN. Skipping."
                            )
                            continue
                        target_summary = {}

                        logger.info(f"Fitting linear model for {output}...")
                        print_summary(f"Linear Regression for '{output}'", [])
                        try:
                            X_df = pd.DataFrame(X, columns=feature_names, index=y.index)
                            model = fit_linear_model(
                                X_df,
                                y,
                                df,
                                output,
                                config["use_wls"],
                                config["significant_only"],
                            )
                            logger.debug(f"Successfully fitted linear model for {output}")
                        except Exception as e:
                            logger.exception(
                                f"Failed to fit linear model for {output}: {e}"
                            )
                            continue

                        try:
                            coef = model.params
                            coef_df = pd.DataFrame({"Coef.": coef})
                            coef_df.to_csv(os.path.join(output_dir, "Models", f"linear_regression_{output}.csv"))
                            logger.debug(f"Saved linear regression coefficients for {output}")
                        except Exception as e:
                            logger.exception(f"Failed to save linear regression coefficients for {output}: {e}")

                        try:
                            coef_table = model.summary2().tables[1]
                            new_index = ["Intercept"] + list(feature_names)
                            coef_table.index = new_index[: len(coef_table)]
                            coef_table.index = clean_linear_terms(coef_table.index)
                            logger.debug(
                                f"Coefficients table for {output}:\n{coef_table}\n"
                            )
                        except Exception as e:
                            logger.exception(f"Failed to create coefficients table for {output}: {e}")
                            continue

                        try:
                            ref_terms = [
                                f"{col} = {config['reference_levels'][col]}"
                                for col in config["input_categoricals"]
                                if col in config.get("reference_levels", {})
                            ]
                            if ref_terms:
                                logger.debug(f"Reference terms for {output}: {ref_terms}")

                            if config["significant_only"]:
                                logger.debug(f"Significant only flag is set for {output}.")

                            print_table(
                                coef_table, title="Significant Coefficients (p < 0.05)"
                            )
                        except Exception as e:
                            logger.exception(f"Failed to process reference terms for {output}: {e}")

                        # Run all available analyses and save plots
                        logger.info(f"Running Random Forest for {output}...")
                        try:
                            rf, scores = fit_random_forest(X, y)
                            print_summary(
                                f"Random Forest CV scores for '{output}'", [str(scores)]
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to fit Random Forest for {output}: {e}"
                            )
                            continue

                        # Save the random forest model
                        try:
                            joblib.dump(rf, os.path.join(output_dir, "Models", f"rf_model_{output}.joblib"))
                            pd.DataFrame(X, columns=feature_names).to_csv(
                                os.path.join(output_dir, "Models", f"X_{output}.csv"), index=False
                            )
                            pd.DataFrame({output: y}).to_csv(
                                os.path.join(output_dir, "Models", f"y_{output}.csv"), index=False
                            )
                            logger.debug(f"Saved random forest model and data for {output}")
                        except Exception as e:
                            logger.exception(f"Failed to save random forest model for {output}: {e}")

                        logger.info(f"Running SHAP analysis for {output}...")
                        try:
                            shap_output_dir = os.path.join(output_dir, "SHAP")
                            mean_shap, top_idx, shap_values = explain_shap(
                                rf,
                                X,
                                feature_names,
                                significant_only=False,
                                save_plots=True,
                                check_additivity=False,
                                output_dir=shap_output_dir, 
                            )
                        except TypeError:
                            try:
                                mean_shap, top_idx, shap_values = explain_shap(
                                    rf,
                                    X,
                                    feature_names,
                                    significant_only=False,
                                    save_plots=True,
                                    output_dir=shap_output_dir,
                                )
                            except Exception as e:
                                logger.exception(f"Failed SHAP analysis for {output}: {e}")
                                continue
                        except Exception as e:
                            logger.exception(f"Failed SHAP analysis for {output}: {e}")
                            continue
                            
                        try:
                            for fname in ["shap_summary.png", "shap_summary_all.png"]:
                                if os.path.exists(fname):
                                    os.replace(fname, os.path.join(output_dir, fname))
                            logger.debug(f"Moved SHAP plots for {output}")
                        except Exception as e:
                            logger.exception(f"Failed to move SHAP plots for {output}: {e}")

                        logger.info(f"Running PDP for {output}...")
                        try:
                            pdp_output_dir = os.path.join(output_dir, "PDP")
                            plot_pdp(
                                rf,
                                X,
                                feature_names,
                                top_idx,
                                significant_only=False,
                                save_plots=True,
                                output_dir=pdp_output_dir,
                            )
                        except Exception as e:
                            logger.exception(f"Failed PDP for {output}: {e}")

                        logger.info(f"Running ANOVA for {output}...")
                        try:
                            anova_output_dir = os.path.join(output_dir, "ANOVA")
                            anova_results = run_anova(
                                df,
                                output,
                                config["input_categoricals"],
                                config["input_numerics"],
                                output_dir=anova_output_dir,
                            )
                            
                            if anova_results:
                                for depth, model in anova_results.items():
                                    anova_table = model.summary2().tables[1]
                                    anova_table.index = clean_anova_terms(anova_table.index)
                                    anova_csv = os.path.join(
                                        output_dir, "ANOVA", f"anova_{output}_depth{depth}.csv"
                                    )
                                    anova_table.to_csv(anova_csv)
                                print_summary(
                                    f"ANOVA results for '{output}'", [str(anova_results)]
                                )
                                logger.debug(f"Successfully completed ANOVA for {output}")
                            else:
                                logger.warning(f"No ANOVA results returned for {output}")
                        except Exception as e:
                            logger.exception(f"Failed ANOVA for {output}: {e}")
                            
                    except Exception as e:
                        logger.exception(
                            f"Exception occurred during analysis for output target '{output}': {e}"
                        )
                        continue
                        
        except Exception as e:
            logger.exception(f"Failed to write results file: {e}")
        finally:
            sys.stdout = orig_stdout

        logger.info("Analysis completed successfully.")
        # Only call generate_auto_summary if it is enabled and target_summaries is populated
        if config.get("generate_summary", False) and target_summaries:
            try:
                generate_auto_summary(target_summaries, config)
            except Exception as e:
                logger.exception(f"Failed to generate auto summary: {e}")
                
    except Exception as e:
        logger.exception(f"Exception occurred during analysis: {e}")
        print(f"Exception occurred: {e}")
