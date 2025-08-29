"""
Modeling utilities for regression and machine learning analysis.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.api import OLS, WLS, add_constant


def fit_linear_model(X, y, df, output, use_wls, significant_only):
    """Fit a linear regression model, with optional WLS and significance filtering.

    Args:
        X: The input features.
        y: The target variable.
        df: The DataFrame containing the data.
        output: The output variable name.
        use_wls: Whether to use weighted least squares.
        significant_only: Whether to filter for significant variables only.

    Returns:
        The fitted model.
    """
    X_sm = add_constant(X)
    if use_wls and output + "Std" in df.columns:
        weights = 1.0 / (df[output + "Std"] ** 2 + 1e-6)
        model = WLS(y, X_sm, weights=weights).fit()
    else:
        model = OLS(y, X_sm).fit()

    return model


def fit_random_forest(X, y):
    """Fit a random forest regressor and evaluate using cross-validation.

    Args:
        X: The input features.
        y: The target variable.

    Returns:
        The fitted random forest model and the cross-validated scores.
    """
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    return rf, scores
