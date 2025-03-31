import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from patsy import dmatrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from statsmodels.formula.api import ols

from itertools import combinations
from math import ceil
from statsmodels.api import WLS, add_constant

def analyze_trends(
    csv_path,
    input_categoricals,
    input_numerics,
    output_targets,
    dropna_required=True,
    use_wls=True
    ):
  
    # Load data and infer column names
    df = pd.read_csv(csv_path)
    print(f"\nLoaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Columns:", list(df.columns))

    # Handle missing values
    if dropna_required:
        all_needed = input_categoricals + input_numerics + output_targets
        missing_summary = df[all_needed].isnull().sum()
        print("\nMissing Values Summary:")
        print(missing_summary[missing_summary > 0])
        df = df.dropna(subset=all_needed)
        print(f"Remaining rows after dropping incomplete cases: {len(df)}")

    input_cols = input_categoricals + input_numerics

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), input_categoricals),
        ('num', StandardScaler(), input_numerics)
    ])
    X = preprocessor.fit_transform(df[input_cols])
    feature_names = preprocessor.get_feature_names_out()

    for output in output_targets:
        print(f"\n{'='*30}\nAnalyzing output: {output}\n{'='*30}")
        y = df[output]

        # ----- LINEAR REGRESSION OR WLS -----
        if use_wls:
            stdev_col = output + "Std"
            if stdev_col in df.columns:
                from statsmodels.api import WLS, add_constant
                weights = 1.0 / (df[stdev_col]**2 + 1e-6)  # avoid divide-by-zero
                X_wls = add_constant(X)
                model_wls = WLS(y, X_wls, weights=weights).fit()
                print("\nWeighted Least Squares Regression (WLS):")
                print(model_wls.summary())
            else:
                print(f"Standard deviation column '{stdev_col}' not found, falling back to OLS.")
                linreg = LinearRegression().fit(X, y)
                print("\nLinear Regression Coefficients:")
                for name, coef in zip(feature_names, linreg.coef_):
                    print(f"{name}: {coef:.4f}")
        else:
            linreg = LinearRegression().fit(X, y)
            print("\nLinear Regression Coefficients:")
            for name, coef in zip(feature_names, linreg.coef_):
                print(f"{name}: {coef:.4f}")

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_

        plt.figure(figsize=(8, 4))
        sns.barplot(x=importances, y=feature_names)
        plt.title(f'Feature Importance for {output} (Random Forest)')
        plt.tight_layout()
        plt.show()

        # SHAP
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names)

        # SHAP Interaction Heatmap
        print("\nSHAP Interaction Heatmap:")
        interaction_values = explainer.shap_interaction_values(X)
        mean_interaction = np.abs(interaction_values).mean(axis=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_interaction,
                    xticklabels=feature_names,
                    yticklabels=feature_names,
                    cmap="viridis",
                    square=True,
                    cbar_kws={"label": "Mean |Interaction Value|"})
        plt.title(f"SHAP Interaction Heatmap for {output}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # PDP grid
        pdp_targets = [(i,) for i in range(X.shape[1])] + list(combinations(range(X.shape[1]), 2))
        total_plots = len(pdp_targets)
        cols = 3
        rows_per_fig = 2
        plots_per_fig = cols * rows_per_fig
        n_figs = ceil(total_plots / plots_per_fig)
        print(f"Generating {total_plots} PDPs across {n_figs} figures.")

        for fig_num in range(n_figs):
            start = fig_num * plots_per_fig
            end = min(start + plots_per_fig, total_plots)
            fig, axes = plt.subplots(rows_per_fig, cols, figsize=(18, 10))
            axes = axes.flatten()

            for ax_idx, pdp_target in enumerate(pdp_targets[start:end]):
                ax = axes[ax_idx]
                try:
                    PartialDependenceDisplay.from_estimator(
                        rf, X, [pdp_target],
                        feature_names=feature_names,
                        kind="average",
                        ax=ax
                    )
                    label = " × ".join([feature_names[i] for i in pdp_target])
                    ax.set_title(f"PDP: {label}")
                except Exception as e:
                    ax.set_visible(False)
                    print(f"Skipped PDP for {pdp_target}: {e}")

            for ax in axes[len(pdp_targets[start:end]):]:
                ax.set_visible(False)

            plt.suptitle(f"Partial Dependence Plots (Page {fig_num + 1})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # ----- FLEXIBLE ANOVA -----
        print("\nANOVA: Base model with main effects")
        terms = []
        for col in input_categoricals:
            terms.append(f'C(Q("{col}"))')
        for col in input_numerics:
            terms.append(f'Q("{col}")')
        base_formula = f'Q("{output}") ~ ' + " + ".join(terms)
        try:
            model = ols(base_formula, data=df).fit()
            print(model.summary())
        except Exception as e:
            print(f"Base ANOVA model failed: {e}")

        # 3-way interaction model
        if len(input_categoricals) >= 2 and len(input_numerics) >= 1:
            print("\nANOVA: 3-way interaction model")
            cat1, cat2 = input_categoricals[:2]
            num1 = input_numerics[0]
            interaction_formula = f'Q("{output}") ~ C(Q("{cat1}")) * C(Q("{cat2}")) * Q("{num1}")'
            try:
                model_inter = ols(interaction_formula, data=df).fit()
                print(model_inter.summary())
            except Exception as e:
                print(f"3-way interaction ANOVA model failed: {e}")
        else:
            print("Skipping 3-way interaction model (need ≥2 categoricals and ≥1 numeric).")