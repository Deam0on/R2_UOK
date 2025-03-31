import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from statsmodels.formula.api import ols

def analyze_trends(csv_path):
    # Load data
    # Load data
    df = pd.read_csv(csv_path)

    # Rename for consistency
    df.columns = ['API', 'Stabiliser', 'MixingChamber', 'Flowrate',
                'MeanNumberSize', 'MeanNumberSizeStd',
                'MeanVolumeSize', 'MeanVolumeSizeStd']

    # --- Handle missing values ---
    missing_summary = df.isnull().sum()
    print("\nMissing Values Summary:")
    print(missing_summary[missing_summary > 0])

    # Drop rows where any input or output is missing
    df = df.dropna(subset=['API', 'Stabiliser', 'MixingChamber', 'Flowrate',
                        'MeanNumberSize', 'MeanVolumeSize'])

    print(f"\nRemaining rows after dropping incomplete cases: {len(df)}")
    
    # Define input and output columns
    categorical_cols = ['API', 'Stabiliser', 'MixingChamber']
    numeric_cols = ['Flowrate']
    input_cols = categorical_cols + numeric_cols
    
    outputs = ['MeanNumberSize', 'MeanVolumeSize']
    
    # Preprocessing: one-hot encode categorical, scale numeric
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])
    
    # Fit-transform input features
    X = preprocessor.fit_transform(df[input_cols])
    feature_names = preprocessor.get_feature_names_out()
    
    for output in outputs:
        print(f"\n{'='*30}\nAnalyzing output: {output}\n{'='*30}")
        y = df[output]

        # ----- LINEAR REGRESSION -----
        print("\nLinear Regression Coefficients:")
        linreg = LinearRegression()
        linreg.fit(X, y)
        for name, coef in zip(feature_names, linreg.coef_):
            print(f"{name}: {coef:.4f}")
        
        # ----- RANDOM FOREST & FEATURE IMPORTANCE -----
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=importances, y=feature_names)
        plt.title(f'Feature Importance for {output} (Random Forest)')
        plt.tight_layout()
        plt.show()

        # ----- SHAP ANALYSIS -----
        explainer = shap.Explainer(rf, X)
        shap_values = explainer(X, check_additivity=False)  # <<< FIX
        shap.summary_plot(shap_values, features=X, feature_names=feature_names)

        # ----- PARTIAL DEPENDENCE (INTERACTIONS) -----
        print("Partial dependence plot for top 2 features:")
        top2_idx = np.argsort(importances)[-2:]
        display = PartialDependenceDisplay.from_estimator(
            rf, X, [tuple(top2_idx)], feature_names=feature_names, kind="average"
        )
        plt.tight_layout()
        plt.show()

        # ----- OPTIONAL: ANOVA for categorical effects -----
        print("\nANOVA Analysis (simplified):")
        formula = f"{output} ~ C(API) + C(Stabiliser) + C(MixingChamber) + Flowrate"
        model = ols(formula, data=df).fit()
        print(model.summary())