# Multi-Parametric Analysis Pipeline

This Python-based pipeline performs multivariate modeling and diagnostics of experimental datasets with categorical and numerical input parameters. It supports a wide range of statistical and machine learning analyses to explore main effects, interactions, and prediction quality.

## Features

- Supports datasets with arbitrary combinations of categorical and numeric inputs and multiple outputs
- Automatic preprocessing (one-hot encoding + scaling)
- Linear regression and optional Weighted Least Squares (WLS)
- Random Forest regression with feature importance and SHAP explanations
- SHAP interaction heatmaps and ranked top interactions
- Partial Dependence Plots (1D and 2D), arranged in 3-column grid layout
- Flexible ANOVA with support for columns with spaces
- Effect size estimation using η²
- Residual analysis: distribution, vs. prediction, and ranking of top errors
- Correlation matrix of numeric inputs
- PCA visualization of explained variance
- Cross-validated R²
- KMeans clustering on categorical inputs

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

```python
analyze_trends(
    csv_path="your_data.csv",
    input_categoricals=["API", "Stabilizers", "Mixing chambers"],
    input_numerics=["Flowrate"],
    output_targets=["Mean num nm", "Mean vol nm"],
    use_wls=True
)
```

## Data Expectations

- Input CSV must contain:
  - One or more categorical input columns
  - One or more numerical input columns
  - One or more output columns
  - Optionally: matching standard deviation columns (e.g., `Mean num nmStd`)
- Column names **can include spaces** (internally quoted using `Q("...")`)

## Interpretation Guide

### Linear Regression / WLS
- Coefficients show additive contributions
- WLS weights samples by inverse variance if std columns are present

### Feature Importance (Random Forest)
- Bar plots rank global relevance of inputs
- Useful for nonlinear importance analysis

### SHAP Summary & Interaction
- SHAP plots show directionality and magnitude of influence
- SHAP interaction heatmaps highlight feature pair effects
- Ranked list of top interaction pairs

### Partial Dependence Plots (PDP)
- 1D plots show marginal influence of each input
- 2D plots capture interaction surfaces
- Arranged in 3-column grid layout per page

### Residual Analysis
- Histogram and scatter of residuals
- Ranking of top 5 mispredicted samples (for outlier detection)

### PCA
- Visual summary of variance explained by input combinations

### ANOVA
- Flexible model building from input specification
- Includes R² and η² (effect size) summaries
- Optional 3-way interaction model auto-configured

### Clustering
- KMeans clusters based on categorical inputs
- Mean outputs compared across clusters

## License

MIT
