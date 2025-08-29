"""
Modeling utilities for regression and machine learning analysis.
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.api import OLS, WLS, add_constant


def fit_linear_model(X, y, df, output, use_wls, significant_only):
    """
    Fit a linear regression model, with optional WLS and significance filtering.

    Args:
        X: The input features.
        y: The target variable.
        df: The DataFrame containing the data.
        output: The output variable name.
        use_wls: Whether to use weighted least squares.
        significant_only: Whether to filter for significant variables only.

    Returns:
        The fitted model.
        
    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If model fitting fails.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if X is None or X.empty:
            raise ValueError("X is None or empty")
        if y is None or len(y) == 0:
            raise ValueError("y is None or empty")
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        
        logger.debug(f"Fitting linear model for {output} with X shape: {X.shape}")
        
        X_sm = add_constant(X)
        
        if use_wls and output + "Std" in df.columns:
            logger.debug("Using weighted least squares (WLS)")
            try:
                weights = 1.0 / (df[output + "Std"] ** 2 + 1e-6)
                model = WLS(y, X_sm, weights=weights).fit()
            except Exception as e:
                logger.exception(f"WLS fitting failed for {output}, falling back to OLS: {e}")
                model = OLS(y, X_sm).fit()
        else:
            logger.debug("Using ordinary least squares (OLS)")
            model = OLS(y, X_sm).fit()
        
        logger.debug(f"Model fitted successfully. R²: {model.rsquared:.4f}")
        
        if significant_only:
            logger.debug("Filtering for significant coefficients only")
            # Note: Actual filtering would be done in the calling function
            # based on model.pvalues < 0.05
            
        return model
        
    except Exception as e:
        logger.exception(f"Failed to fit linear model for {output}: {e}")
        raise RuntimeError(f"Linear model fitting failed: {e}") from e


def fit_random_forest(X, y):
    """
    Fit a random forest regressor and evaluate using cross-validation.

    Args:
        X: The input features.
        y: The target variable.

    Returns:
        The fitted random forest model and the cross-validated scores.
        
    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If model fitting fails.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if X is None:
            raise ValueError("X is None")
        if y is None or len(y) == 0:
            raise ValueError("y is None or empty")
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        
        logger.debug(f"Fitting Random Forest with X shape: {X.shape}")
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        logger.debug("Computing cross-validation scores")
        scores = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
        
        logger.debug(f"Random Forest fitted successfully. CV R² mean: {scores.mean():.4f} (±{scores.std():.4f})")
        
        return rf, scores
        
    except Exception as e:
        logger.exception(f"Failed to fit Random Forest: {e}")
        raise RuntimeError(f"Random Forest fitting failed: {e}") from e
