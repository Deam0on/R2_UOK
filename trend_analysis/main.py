### main.py (modified)

from trend_analysis.config import config as default_config
import logging
import numpy as np
import sys
from trend_analysis.config import config as default_config
from trend_analysis.utils import (
    print_table, print_summary, format_feature_name,
    print_top_features, print_model_metrics, setup_logger, check_imbalance,
    transform_skewed_columns, clean_linear_terms, clean_anova_terms, generate_auto_summary
)
from trend_analysis.preprocess import load_and_clean, build_preprocessor
from trend_analysis.visualization import show_correlation, plot_pca
from trend_analysis.modeling import fit_linear_model, fit_random_forest
from trend_analysis.shap_analysis import explain_shap
from trend_analysis.pdp_analysis import plot_pdp
from trend_analysis.anova import run_anova
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

def main(config=None):
    print("[DEBUG] Entered main()")
    # Enhanced logging setup: file and console
    setup_logger()
    logger = logging.getLogger()
    # Remove all handlers associated with the root logger object (avoid duplicate logs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Add file handler
    file_handler = logging.FileHandler('analysis.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    logger.info("Starting trend analysis main function.")
    try:
        if config is None:
            raise ValueError("No config provided to main(). Please use the CLI or pass a config dict.")

        df = load_and_clean(
            config.get("csv_path"),
            config["input_categoricals"] + config["input_numerics"],
            config["output_targets"],
            dropna_required=config.get("dropna_required", True),
            config=config
        )
        logger.info(f"Loaded dataframe with shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"First 5 rows:\n{df.head()}\n")
        import os
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'analysis_results.txt')
        # Save head of dataframe to results file and as CSV
        df.head().to_csv(os.path.join(output_dir, 'data_head.csv'), index=False)
        with open(results_path, 'w', encoding='utf-8') as results_file:
            results_file.write('First 5 rows of data:\n')
            results_file.write(df.head().to_string())
            results_file.write('\n\n')

        if not config.get("dropna_required", False):
            logger.info("Imputing missing values for numeric columns.")
            imputer = SimpleImputer(strategy="mean")
            df[config["input_numerics"]] = imputer.fit_transform(df[config["input_numerics"]])

        logger.info("Processing categorical reference levels...")
        for col in config["input_categoricals"]:
            ref = str(config.get("reference_levels", {}).get(col))
            logger.debug(f"Reference for {col}: {ref}")

        if config.get("run_imbalance_check", False):
            logger.info("Checking for feature imbalance...")
            skewed_cols = check_imbalance(df, config)
            logger.debug(f"Skewed columns: {skewed_cols}")

        logger.info("Building preprocessor and transforming features...")
        preprocessor = build_preprocessor(config["input_categoricals"], config["input_numerics"])
        X = preprocessor.fit_transform(df[config["input_categoricals"] + config["input_numerics"]])
        feature_names = preprocessor.get_feature_names_out()
        logger.debug(f"Feature names after preprocessing: {feature_names}")

        logger.info("Plotting correlation matrix...")
        show_correlation(df, config["input_numerics"], save_plots=True)
        # Move correlation_matrix.png to output dir if it exists
        if os.path.exists('correlation_matrix.png'):
            os.replace('correlation_matrix.png', os.path.join(output_dir, 'correlation_matrix.png'))
        logger.info("Plotting PCA...")
        plot_pca(X, PCA(), save_plots=True)
        if os.path.exists('pca_variance.png'):
            os.replace('pca_variance.png', os.path.join(output_dir, 'pca_variance.png'))

        target_summaries = {}

        # Redirect all print output to results file
        import io
        orig_stdout = sys.stdout
        with open(results_path, 'a', encoding='utf-8') as results_file:
            sys.stdout = results_file
            try:
                for output in config["output_targets"]:
                    try:
                        logger.info(f"Processing output target: {output}")
                        if output not in df.columns:
                            logger.error(f"Output target '{output}' not found in dataframe columns. Skipping.")
                            continue
                        y = df[output]
                        if y.isnull().all():
                            logger.error(f"All values for output target '{output}' are NaN. Skipping.")
                            continue
                        target_summary = {}

                        logger.info(f"Fitting linear model for {output}...")
                        print_summary(f"Linear Regression for '{output}'", [])
                        try:
                            model = fit_linear_model(X, y, df, output, config["use_wls"], config["significant_only"])
                        except Exception as e:
                            logger.exception(f"Failed to fit linear model for {output}: {e}")
                            continue

                        coef_table = model.summary2().tables[1]
                        new_index = ["Intercept"] + list(feature_names)
                        coef_table.index = new_index[: len(coef_table)]
                        coef_table.index = clean_linear_terms(coef_table.index)
                        logger.debug(f"Coefficients table for {output}:\n{coef_table}\n")

                        ref_terms = [f"{col} = {config['reference_levels'][col]}" for col in config["input_categoricals"] if col in config.get("reference_levels", {})]
                        if ref_terms:
                            logger.debug(f"Reference terms for {output}: {ref_terms}")

                        if config["significant_only"]:
                            logger.debug(f"Significant only flag is set for {output}.")

                        print_table(coef_table, title="Significant Coefficients (p < 0.05)")

                        # Run all available analyses and save plots
                        logger.info(f"Running Random Forest for {output}...")
                        try:
                            rf, scores = fit_random_forest(X, y)
                            print_summary(f"Random Forest CV scores for '{output}'", [str(scores)])
                        except Exception as e:
                            logger.exception(f"Failed to fit Random Forest for {output}: {e}")
                            continue

                        logger.info(f"Running SHAP analysis for {output}...")
                        try:
                            from trend_analysis.shap_analysis import explain_shap
                            mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, significant_only=False, save_plots=True, check_additivity=False)
                        except TypeError:
                            mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, significant_only=False, save_plots=True)
                        except Exception as e:
                            logger.exception(f"Failed SHAP analysis for {output}: {e}")
                            continue
                        for fname in ["shap_summary.png", "shap_summary_all.png"]:
                            if os.path.exists(fname):
                                os.replace(fname, os.path.join(output_dir, fname))

                        logger.info(f"Running PDP for {output}...")
                        try:
                            plot_pdp(rf, X, feature_names, top_idx, significant_only=False, save_plots=True)
                        except Exception as e:
                            logger.exception(f"Failed PDP for {output}: {e}")

                        logger.info(f"Running ANOVA for {output}...")
                        try:
                            anova_results = run_anova(df, output, config["input_categoricals"], config["input_numerics"])
                            from trend_analysis.utils import clean_anova_terms
                            for depth, model in anova_results.items():
                                anova_table = model.summary2().tables[1]
                                anova_table.index = clean_anova_terms(anova_table.index)
                                anova_csv = os.path.join(output_dir, f'anova_{output}_depth{depth}.csv')
                                anova_table.to_csv(anova_csv)
                            print_summary(f"ANOVA results for '{output}'", [str(anova_results)])
                        except Exception as e:
                            logger.exception(f"Failed ANOVA for {output}: {e}")
                    except Exception as e:
                        logger.exception(f"Exception occurred during analysis for output target '{output}': {e}")
                        continue
            finally:
                sys.stdout = orig_stdout

        logger.info("Analysis completed successfully.")
        # Only call generate_auto_summary if it is enabled and target_summaries is populated
        if config.get("generate_summary", False) and target_summaries:
            generate_auto_summary(target_summaries, config)
    except Exception as e:
        logger.exception(f"Exception occurred during analysis: {e}")
        print(f"Exception occurred: {e}")

        if config.get("run_anova", False):
            anova_models = run_anova(df, output, config["input_categoricals"], config["input_numerics"])
            all_anova = []
            for depth, model in anova_models.items():
                anova_table = model.summary2().tables[1]
                if config["significant_only"]:
                    anova_table = anova_table[anova_table["P>|t|"] < 0.05]
                anova_table.index = clean_anova_terms(anova_table.index)
                if ref_terms:
                    if "Intercept" in anova_table.index:
                        anova_table.rename(index={"Intercept": f"Intercept ({', '.join(ref_terms)})"}, inplace=True)
                print_table(anova_table, title=f"ANOVA: Depth {depth} Significant Terms (p < 0.05)")
                if depth == 1:
                    target_summary["anova_table"] = anova_table

        target_summaries[output] = target_summary

    if config.get("generate_summary", False):
        generate_auto_summary(target_summaries, config)