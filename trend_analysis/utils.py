### utils.py (updated)

from tabulate import tabulate
import logging
from  sklearn import metrics
from scipy.stats import skew
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import re
from tabulate import tabulate
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def generate_auto_summary(target_summaries, config):
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
        intercept_label = [idx for idx in data["coef_table"].index if idx.lower().startswith("intercept")]
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
        if any((data["anova_table"]["P>|t|"] > 0.05) & (data["anova_table"]["P>|t|"] < 0.1)):
            print("  Note: Some terms were marginally significant (0.05 < p < 0.1)")

        ### Model Fit Summary
        print("\n- Model Performance:")
        print(f"  R² = {data['metrics']['r2']:.3f}, MAE = {data['metrics']['mae']:.2f}, RMSE = {data['metrics']['rmse']:.2f}")
        if "cv_scores" in data and data["cv_scores"] is not None:
            print(f"  Random Forest CV R² = {np.mean(data['cv_scores']):.3f} ± {np.std(data['cv_scores']):.3f}")

        ### Skewness Note
        for col, val in data.get("skew_info", {}).items():
            if abs(val) > 1:
                print(f"  Note: {col} was highly skewed (skew = {val:.2f}) and may have been transformed.")

        print("\n------------------------------------\n")

def print_table(df, title=None, floatfmt=".4f"):
    if title:
        print(f"\n{title}")
    print(tabulate(df, headers="keys", floatfmt=floatfmt, tablefmt="pretty"))

def print_summary(title, lines):
    print(f"\n{'='*len(title)}\n{title}\n{'='*len(title)}")
    for line in lines:
        print(f"  - {line}")

def format_feature_name(name):
    return name.replace(":", " × ").replace("_", " ").title()

def print_top_features(shap_values, feature_names, top_n=5):
    mean_shap = shap_values.mean(axis=0)
    top_idx = mean_shap.argsort()[-top_n:][::-1]
    print(f"\nTop {top_n} most impactful features by SHAP:")
    for rank, idx in enumerate(top_idx, 1):
        fname = format_feature_name(feature_names[idx])
        print(f"{rank}. {fname} (mean SHAP = {mean_shap[idx]:.4f})")
    return top_idx

def clean_linear_terms(index_list):
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
    print_summary("Model Evaluation", [
        f"R² = {r2_score(y_true, y_pred):.3f}",
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}",
        f"RMSE = {root_mean_squared_error(y_true, y_pred):.3f}"
    ])

def setup_logger(logfile="analysis.log", level=logging.INFO):
    logging.basicConfig(
        filename=logfile,
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_imbalance(df, config, skew_threshold=1.0):
    print_summary("Checking for feature imbalances", [])
    skewed_cols = []

    for col in config["input_numerics"]:
        s = skew(df[col].dropna())
        if abs(s) > skew_threshold:
            print(f"{col} is highly skewed (skew = {s:.2f})")
            skewed_cols.append(col)

    for target in config["output_targets"]:
        s = skew(df[target].dropna())
        if abs(s) > skew_threshold:
            print(f"Output {col} is highly skewed (skew = {s:.2f})")
            skewed_cols.append(col)

    return skewed_cols

def clean_anova_terms(index_list):
    cleaned = []
    for i in index_list:
        if "Intercept" in i:
            cleaned.append("Intercept")
            continue

        # Replace Q("X") → X
        i = re.sub(r'Q\("([^"]+)"\)', r'\1', i)

        # Replace C("X") → X
        i = re.sub(r'C\("([^"]+)"\)', r'\1', i)

        # Replace [T.X] or [X] → = X
        i = re.sub(r'\[T\.(.*?)\]', r'= \1', i)
        i = re.sub(r'\[(.*?)\]', r'= \1', i)

        # Replace interaction colons with ×
        i = i.replace(":", " × ")

        cleaned.append(i)

    return cleaned

def transform_skewed_columns(df, config, skew_threshold=1.0):
    """
    Detect and transform highly skewed numeric columns using Yeo-Johnson.
    Returns a new DataFrame with transformed columns.
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


