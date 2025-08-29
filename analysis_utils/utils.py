"""
General utility functions for analysis pipeline, logging, and output formatting.
"""

import logging
import re

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import PowerTransformer
from tabulate import tabulate


def generate_auto_summary(target_summaries, config):
    """
    Generate and print an auto-summary for the target variables in the analysis.

    Args:
        target_summaries (dict): A dictionary containing summary data for each target variable.
        config (dict): Configuration settings for the analysis, including reference levels.
    """
    print("\n===== AUTO-GENERATED SUMMARY =====\n")

    # Print reference levels if available
    ref_levels = config.get("reference_levels", {})
    if ref_levels:
        print("Reference Levels:")
        for col, ref in ref_levels.items():
            print(f"- {col} = {ref}")
        print()  # Blank line for spacing

    for target, data in target_summaries.items():
        print(f"### Summary for: {target}\n")

        ### SHAP Summary
        print("- SHAP-Based Feature Impact:")
        for feature, value, direction in data["shap_top"]:
            print(f"  {feature} had a {direction} impact (mean SHAP = {value:.2f})")

        ### Regression Coeff Summary
        print("\n- Linear Model Interpretation:")
        intercept_label = [
            idx
            for idx in data["coef_table"].index
            if idx.lower().startswith("intercept")
        ]
        if intercept_label:
            intercept = data["coef_table"].loc[intercept_label[0], "Coef."]
            print(f"  Baseline prediction (intercept): {intercept:.2f}")
        for i, row in data["coef_table"].iterrows():
            if "Intercept" in i:
                continue
            coef = row["Coef."]
            effect = "increases" if coef > 0 else "decreases"
            print(f"  {i} {effect} the output by {abs(coef):.2f}")

        ### ANOVA Summary
        print("\n- Statistically Significant Effects (ANOVA):")
        sig_terms = data["anova_table"].loc[data["anova_table"]["P>|t|"] < 0.05]
        for i, row in sig_terms.iterrows():
            if "F" in row:
                effect = f"{i} (p = {row['P>|t|']:.3f}, F = {row['F']:.2f})"
            else:
                effect = f"{i} (p = {row['P>|t|']:.3f})"
            print(effect)
        if any(
            (data["anova_table"]["P>|t|"] > 0.05) & (data["anova_table"]["P>|t|"] < 0.1)
        ):
            print("  Note: Some terms were marginally significant (0.05 < p < 0.1)")

        ### Model Fit Summary
        print("\n- Model Performance:")
        print(
            f"  R² = {data['metrics']['r2']:.3f}, MAE = {data['metrics']['mae']:.2f}, RMSE = {data['metrics']['rmse']:.2f}"
        )
        if "cv_scores" in data and data["cv_scores"] is not None:
            print(
                f"  Random Forest CV R² = {np.mean(data['cv_scores']):.3f} ± {np.std(data['cv_scores']):.3f}"
            )

        ### Skewness Note
        for col, val in data.get("skew_info", {}).items():
            if abs(val) > 1:
                print(
                    f"  Note: {col} was highly skewed (skew = {val:.2f}) and may have been transformed."
                )

        print("\n------------------------------------\n")


def print_table(df, title=None, floatfmt=".4f"):
    """
    Print a DataFrame as a formatted table.

    Args:
        df (DataFrame): The DataFrame to print.
        title (str, optional): An optional title for the table.
        floatfmt (str, optional): Format string for floating-point numbers.
    """
    if title:
        print(f"\n{title}")
    print(tabulate(df, headers="keys", floatfmt=floatfmt, tablefmt="pretty"))


def print_summary(title, lines):
    """
    Print a formatted summary section.

    Args:
        title (str): The title of the summary section.
        lines (list): A list of lines to include in the summary.
    """
    print(f"\n{'='*len(title)}\n{title}\n{'='*len(title)}")
    for line in lines:
        print(f"  - {line}")


def format_feature_name(name):
    """
    Format a feature name for display.

    Args:
        name (str): The raw feature name.

    Returns:
        str: The formatted feature name.
    """
    return name.replace(":", " × ").replace("_", " ").title()


def print_top_features(shap_values, feature_names, top_n=5):
    """
    Print the top N most impactful features based on SHAP values.

    Args:
        shap_values (array): The SHAP values for the features.
        feature_names (array): The names of the features.
        top_n (int, optional): The number of top features to display.

    Returns:
        array: Indices of the top features.
    """
    mean_shap = shap_values.mean(axis=0)
    top_idx = mean_shap.argsort()[-top_n:][::-1]
    print(f"\nTop {top_n} most impactful features by SHAP:")
    for rank, idx in enumerate(top_idx, 1):
        fname = format_feature_name(feature_names[idx])
        print(f"{rank}. {fname} (mean SHAP = {mean_shap[idx]:.4f})")
    return top_idx


def clean_linear_terms(index_list):
    """
    Clean and format linear term names for display.

    Args:
        index_list (list): The list of term names (e.g., from a regression model).

    Returns:
        list: The cleaned and formatted term names.
    """
    cleaned = []
    for name in index_list:
        if name == "Intercept":
            cleaned.append(name)
        elif name.startswith("cat__"):
            parts = name.split("__")
            # Handle typical 'cat__Col__Value' pattern
            if len(parts) >= 3:
                cleaned.append(f"{parts[1]} = {parts[2]}")
            elif len(parts) > 3:
                cleaned.append(f"{parts[1]} = {'__'.join(parts[2:])}")
            else:
                cleaned.append(name)  # fallback
        else:
            cleaned.append(name)
    return cleaned


def print_model_metrics(y_true, y_pred):
    """
    Print model evaluation metrics: R², MAE, and RMSE.

    Args:
        y_true (array): True target values.
        y_pred (array): Predicted values from the model.
    """
    print_summary(
        "Model Evaluation",
        [
            f"R² = {r2_score(y_true, y_pred):.3f}",
            f"MAE = {mean_absolute_error(y_true, y_pred):.3f}",
            f"RMSE = {root_mean_squared_error(y_true, y_pred):.3f}",
        ],
    )


def setup_logger(logfile="analysis.log", level=logging.INFO):
    """
    Set up the logging configuration for the analysis.

    Args:
        logfile (str, optional): The log file name.
        level (int, optional): The logging level (e.g., logging.INFO).
    """
    logging.basicConfig(
        filename=logfile,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_imbalance(df, config, skew_threshold=1.0):
    """
    Check for feature and target variable imbalance based on skewness.

    Args:
        df (DataFrame): The input DataFrame containing features and targets.
        config (dict): Configuration settings, including input and output variable lists.
        skew_threshold (float, optional): The skewness threshold for identifying imbalance.

    Returns:
        list: A list of columns that are highly skewed.
        
    Raises:
        ValueError: If invalid inputs are provided.
    """
    logger = logging.getLogger(__name__)
    
    try:
        print_summary("Checking for feature imbalances", [])
        skewed_cols = []

        # Check numeric input features
        for col in config.get("input_numerics", []):
            try:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in DataFrame")
                    continue
                    
                s = skew(df[col].dropna())
                if abs(s) > skew_threshold:
                    print(f"{col} is highly skewed (skew = {s:.2f})")
                    logger.info(f"Feature {col} is highly skewed (skew = {s:.2f})")
                    skewed_cols.append(col)
            except Exception as e:
                logger.exception(f"Failed to check skewness for input feature '{col}': {e}")

        # Check output targets - fixed variable name bug
        for target in config.get("output_targets", []):
            try:
                if target not in df.columns:
                    logger.warning(f"Target '{target}' not found in DataFrame")
                    continue
                    
                s = skew(df[target].dropna())
                if abs(s) > skew_threshold:
                    print(f"Output {target} is highly skewed (skew = {s:.2f})")
                    logger.info(f"Output target {target} is highly skewed (skew = {s:.2f})")
                    skewed_cols.append(target)
            except Exception as e:
                logger.exception(f"Failed to check skewness for output target '{target}': {e}")

        logger.info(f"Found {len(skewed_cols)} highly skewed columns")
        return skewed_cols
        
    except Exception as e:
        logger.exception(f"Failed to check imbalance: {e}")
        return []


def clean_anova_terms(index_list):
    """
    Clean and format ANOVA term names for display.

    Args:
        index_list (list): The list of ANOVA term names.

    Returns:
        list: The cleaned and formatted term names.
    """
    cleaned = []
    for i in index_list:
        if "Intercept" in i:
            cleaned.append("Intercept")
            continue

        # Replace Q("X") → X
        i = re.sub(r'Q\("([^"]+)"\)', r"\1", i)

        # Replace C("X") → X
        i = re.sub(r'C\("([^"]+)"\)', r"\1", i)

        # Replace [T.X] or [X] → = X
        i = re.sub(r"\[T\.(.*?)\]", r"= \1", i)
        i = re.sub(r"\[(.*?)\]", r"= \1", i)

        # Replace interaction colons with ×
        i = i.replace(":", " × ")

        cleaned.append(i)

    return cleaned


def transform_skewed_columns(df, config, skew_threshold=1.0):
    """
    Detect and transform highly skewed numeric columns using Yeo-Johnson.

    Args:
        df (DataFrame): The input DataFrame containing features and targets.
        config (dict): Configuration settings, including input and output variable lists.
        skew_threshold (float, optional): The skewness threshold for identifying imbalance.

    Returns:
        DataFrame: A new DataFrame with transformed columns.
    """
    pt = PowerTransformer(method="yeo-johnson")
    numeric_cols = config["input_numerics"] + config["output_targets"]

    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = skew(df[col].dropna())
        if abs(s) > skew_threshold:
            print(f"Transforming '{col}' (skew = {s:.2f}) using Yeo-Johnson")
            transformed = pt.fit_transform(df[[col]])
            df[col + "_transformed"] = transformed.flatten()

    return df
