import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.api import WLS, OLS, add_constant

def fit_linear_model(X, y, df, output, use_wls, significant_only):
    X_sm = add_constant(X)
    if use_wls and output + "Std" in df.columns:
        weights = 1.0 / (df[output + "Std"]**2 + 1e-6)
        model = WLS(y, X_sm, weights=weights).fit()
    else:
        model = OLS(y, X_sm).fit()

    return model

def fit_random_forest(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    return rf, scores
