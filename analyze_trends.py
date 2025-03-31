import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from patsy import dmatrix
from statsmodels.api import WLS, add_constant

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

from itertools import combinations
from math import ceil

def analyze_trends(
    csv_path,
    input_categoricals,
    input_numerics,
    output_targets,
    dropna_required=True,
    use_wls=True
    ):

    df = pd.read_csv(csv_path)
    print(f"\nLoaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Columns:", list(df.columns))

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

    # Suggestion 2: Correlation matrix
    if input_numerics:
        print("\nCorrelation Matrix of Numeric Inputs:")
        sns.heatmap(df[input_numerics].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix (Numeric Inputs)")
        plt.tight_layout()
        plt.show()

    # Suggestion 4: PCA
    if X.shape[1] >= 2:
        pca = PCA()
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.title("Cumulative Explained Variance (PCA)")
        plt.xlabel("Number of Components")
        plt.ylabel("Explained Variance")
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Suggestion 5: Clustering (Categorical one-hot only)
    if input_categoricals:
        onehot = OneHotEncoder(sparse_output=False, drop='first')
        X_cat = onehot.fit_transform(df[input_categoricals])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_cat)
        df["Cluster"] = kmeans.labels_
        print("\nCluster Memberships based on Categorical Inputs:")
        print(df.groupby("Cluster")[output_targets].mean())

    for output in output_targets:
        print(f"\n{'='*30}\nAnalyzing output: {output}\n{'='*30}")
        y = df[output]

        # ----- LINEAR REGRESSION OR WLS -----
        if use_wls:
            stdev_col = output + "Std"
            if stdev_col in df.columns:
                weights = 1.0 / (df[stdev_col]**2 + 1e-6)
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

        # Suggestion 3: Cross-validation
        scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
        print(f"\nCross-validated R²: {scores.mean():.3f} ± {scores.std():.3f}")

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

        # Suggestion 7: Top SHAP interactions
        triu_idx = np.triu_indices_from(mean_interaction, k=1)
        top_pairs = np.dstack((triu_idx[0], triu_idx[1]))[0]
        top_sorted = sorted(top_pairs, key=lambda x: mean_interaction[x[0], x[1]], reverse=True)[:5]
        print("\nTop SHAP Feature Interactions:")
        for i, j in top_sorted:
            print(f"{feature_names[i]} × {feature_names[j]}: {mean_interaction[i, j]:.4f}")

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

        # Residual Analysis
        residuals = y - rf.predict(X)
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")
        plt.xlabel("Residual")
        plt.tight_layout()
        plt.show()

        sns.scatterplot(x=rf.predict(X), y=residuals)
        plt.axhline(0, linestyle='--', color='gray')
        plt.title("Residuals vs. Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.show()

        # Flexible ANOVA
        print("\nANOVA: Base model with main effects")
        terms = [f'C(Q("{col}"))' for col in input_categoricals] + [f'Q("{col}")' for col in input_numerics]
        base_formula = f'Q("{output}") ~ ' + " + ".join(terms)
        try:
            model = ols(base_formula, data=df).fit()
            print(model.summary())

            # Suggestion 6: Effect Size (η²)
            ss_total = np.sum((y - np.mean(y))**2)
            ss_model = np.sum((model.fittedvalues - np.mean(y))**2)
            eta_sq = ss_model / ss_total
            print(f"Effect size η² (model R²): {eta_sq:.3f}")
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
