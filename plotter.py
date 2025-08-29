import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Number of top features to plot (change as needed)
TOP_N = 20
output_dir = "output"

# 1. SHAP importances
shap_imp = os.path.join(output_dir, "shap_importance.csv")
if os.path.exists(shap_imp):
    df = pd.read_csv(shap_imp)
    df = df.sort_values("mean_abs_shap", ascending=False).head(TOP_N)
    plt.figure(figsize=(10, 5))
    plt.bar(df["feature"], df["mean_abs_shap"])
    plt.xticks(rotation=90)
    plt.title(f"Top {TOP_N} SHAP Feature Importances")
    plt.tight_layout()
    plt.show()

# 2. 1D PDPs
for pdp_file in glob.glob(os.path.join(output_dir, "pdp_*.csv")):
    # Skip 2D PDPs
    if "_" in os.path.basename(pdp_file)[4:-4]:
        continue
    df = pd.read_csv(pdp_file)
    feature = df.columns[0]
    plt.figure()
    plt.plot(df[feature], df["partial_dependence"])
    plt.xlabel(feature)
    plt.ylabel("Partial Dependence")
    plt.title(f"PDP for {feature}")
    plt.tight_layout()
    plt.show()

# 3. 2D PDPs
for pdp2d_file in glob.glob(os.path.join(output_dir, "pdp_*_*.csv")):
    df = pd.read_csv(pdp2d_file, index_col=0)
    plt.figure()
    sns.heatmap(df, cmap="viridis")
    plt.title(f"2D PDP: {df.index.name} vs {df.columns.name or df.columns[0]}")
    plt.xlabel(df.columns.name or df.columns[0])
    plt.ylabel(df.index.name)
    plt.tight_layout()
    plt.show()

# 4. Linear regression coefficients
for reg_file in glob.glob(os.path.join(output_dir, "linear_regression_*.csv")):
    df = pd.read_csv(reg_file, index_col=0)
    if "Intercept" in df.index:
        df = df.drop("Intercept")
    df = df.sort_values("Coef.", key=abs, ascending=False).head(TOP_N)
    plt.figure()
    df["Coef."].plot(kind="bar")
    plt.title(f"Top {TOP_N} Regression Coefficients: {os.path.basename(reg_file)}")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.show()

# 5. ANOVA tables
for anova_file in glob.glob(os.path.join(output_dir, "anova_*_depth*.csv")):
    df = pd.read_csv(anova_file, index_col=0)
    # Only plot significant terms
    if "P>|t|" in df.columns:
        sig = df[df["P>|t|"] < 0.05]
        sig = sig.sort_values("Coef.", key=abs, ascending=False).head(TOP_N)
        if not sig.empty:
            plt.figure()
            sig["Coef."].plot(kind="bar")
            plt.title(f"Top {TOP_N} ANOVA Significant Coefficients: {os.path.basename(anova_file)}")
            plt.ylabel("Coefficient")
            plt.tight_layout()
            plt.show()

# 6. Correlation matrix
corr_file = os.path.join(output_dir, "correlation_matrix.csv")
if os.path.exists(corr_file):
    df = pd.read_csv(corr_file, index_col=0)
    plt.figure()
    sns.heatmap(df, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()