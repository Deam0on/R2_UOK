import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

output_dir = "output"

# --- 1. SHAP: Group by parameter (using config names) ---
import yaml
config_path = os.path.join(os.path.dirname(__file__), 'trend_analysis', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
param_names = config['input_categoricals'] + config['input_numerics']

shap_imp_path = os.path.join(output_dir, "shap_importance.csv")
if os.path.exists(shap_imp_path):
    df = pd.read_csv(shap_imp_path)
    def get_param(col):
        for p in param_names:
            if p in col:
                return p
        return None
    df["parameter"] = df["feature"].apply(get_param)
    grouped = df.groupby("parameter")["mean_abs_shap"].sum().reindex(param_names).dropna().sort_values(ascending=False)
    print("\nSHAP Parameter Importances (sum of mean_abs_shap):\n", grouped)
    plt.figure(figsize=(8,4))
    grouped.head(20).plot(kind="bar")
    plt.title("Top Parameters by SHAP Importance")
    plt.ylabel("Summed mean_abs_shap")
    plt.tight_layout()
    plt.show()
else:
    print("No SHAP importances found.")

# --- 2. ANOVA: Group by parameter (using config names) ---
anova_files = glob.glob(os.path.join(output_dir, "anova_*_depth1.csv"))
if anova_files:
    for anova_file in anova_files:
        df = pd.read_csv(anova_file, index_col=0)
        # Only use significant terms
        if "P>|t|" in df.columns:
            sig = df[df["P>|t|"] < 0.05]
            def get_param(idx):
                for p in param_names:
                    if p in idx:
                        return p
                return None
            sig["parameter"] = sig.index.map(get_param)
            grouped = sig.groupby("parameter")["Coef."].apply(lambda x: x.abs().sum()).reindex(param_names).dropna().sort_values(ascending=False)
            print(f"\nANOVA Parameter Importances for {os.path.basename(anova_file)}:\n", grouped)
            plt.figure(figsize=(8,4))
            grouped.head(20).plot(kind="bar")
            plt.title(f"Top Parameters by ANOVA (|sum Coef|): {os.path.basename(anova_file)}")
            plt.ylabel("Sum |Coef| (significant)")
            plt.tight_layout()
            plt.show()
else:
    print("No ANOVA files found.")

# --- 3. Regression: Group by parameter (using config names) ---
reg_files = glob.glob(os.path.join(output_dir, "linear_regression_*.csv"))
if reg_files:
    for reg_file in reg_files:
        df = pd.read_csv(reg_file, index_col=0)
        if "Intercept" in df.index:
            df = df.drop("Intercept")
        def get_param(idx):
            for p in param_names:
                if p in idx:
                    return p
            return None
        df["parameter"] = df.index.map(get_param)
        grouped = df.groupby("parameter")["Coef."].apply(lambda x: x.abs().sum()).reindex(param_names).dropna().sort_values(ascending=False)
        print(f"\nRegression Parameter Importances for {os.path.basename(reg_file)}:\n", grouped)
        plt.figure(figsize=(8,4))
        grouped.head(20).plot(kind="bar")
        plt.title(f"Top Parameters by Regression (|sum Coef|): {os.path.basename(reg_file)}")
        plt.ylabel("Sum |Coef|")
        plt.tight_layout()
        plt.show()
else:
    print("No regression coefficient files found.")

# --- 4. Permutation Importance (if data/model available) ---
try:
    import joblib
    from sklearn.inspection import permutation_importance
    # Try to load model and data if available
    model_path = os.path.join(output_dir, "rf_model.joblib")
    X_path = os.path.join(output_dir, "X.csv")
    y_path = os.path.join(output_dir, "y.csv")
    if os.path.exists(model_path) and os.path.exists(X_path) and os.path.exists(y_path):
        model = joblib.load(model_path)
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).squeeze()
        result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
        importances = pd.Series(result.importances_mean, index=X.columns)
        # Group by parameter
        importances_grouped = importances.groupby(importances.index.map(lambda x: x.split("__")[1] if "__" in x else x.split("_")[0])).sum()
        print("\nPermutation Importances (grouped):\n", importances_grouped.sort_values(ascending=False))
        plt.figure(figsize=(8,4))
        importances_grouped.sort_values(ascending=False).head(20).plot(kind="bar")
        plt.title("Top Parameters by Permutation Importance")
        plt.ylabel("Mean Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("Permutation importance: model or data not found in output/.")
except ImportError:
    print("joblib or sklearn not installed; skipping permutation importance.")

print("\nSummary complete.")
