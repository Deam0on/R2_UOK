# Multi-Parametric Analysis Pipeline

This repository provides a Python-based pipeline to analyze the effect of multiple categorical and numerical input variables on one or more numerical outputs. The pipeline includes preprocessing, statistical modeling, feature importance analysis, and interaction visualization.

## Features

- Support for datasets with arbitrary numbers of inputs/outputs
- Automatic preprocessing of categorical and numerical features
- Linear regression and Weighted Least Squares (WLS) modeling (optional)
- Random Forest regression
- SHAP value-based feature importance and interaction analysis
- Partial dependence plots (1D and 2D)
- Flexible ANOVA modeling with support for column names containing spaces

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## How to Use

Call the main function:

```python
analyze_trends(
    csv_path="your_data.csv",
    input_categoricals=[],
    input_numerics=[],
    output_targets=[],
    use_wls=True  # Set to False to disable weighted regression
)
```

## Data Format

The input CSV must contain:
- One or more **categorical input columns** 
- One or more **numerical input columns** 
- One or more **numerical output columns** 
- Optionally: standard deviation columns for outputs 

Standard deviation columns are used only if `use_wls=True`.

## Analysis Performed

### 1. Linear Regression / WLS

- Ordinary least squares (OLS) or weighted least squares (WLS)
- Coefficients quantify the additive contribution of each feature
- WLS uses inverse variance as weight if standard deviations are provided

### 2. Random Forest Regression

- Nonlinear, nonparametric model to capture complex interactions
- Feature importance scores calculated via impurity-based reduction

### 3. SHAP Analysis

- SHAP (SHapley Additive exPlanations) values for feature-level attribution
- Summary plots showing directionality and magnitude of feature effects
- SHAP interaction heatmaps highlight feature pair interactions

### 4. Partial Dependence Plots (PDPs)

- 1D PDPs show the marginal effect of each feature
- 2D PDPs visualize interactions between feature pairs
- Plots arranged in paginated 3-column grid layout

### 5. ANOVA (Analysis of Variance)

- Linear models fitted using `statsmodels` with categorical encoding
- Flexible generation of main-effects models from input specification
- Optional 3-way interaction model (if ≥2 categoricals and ≥1 numeric)
- Full model summaries include R², F-statistic, and p-values

## Interpretation Guide

### Linear Regression / WLS
- Coefficients reflect **direct, additive effects**.
- Sign (+/–) indicates direction; magnitude shows effect size.
- In WLS, larger weights are given to samples with lower standard deviation (more reliable data).

### Feature Importance (Random Forest)
- Tallest bars = most impactful features.
- Good for assessing **nonlinear contributions**, but not direction of effect.

### SHAP Summary Plots
- Show **how individual feature values push predictions up or down**.
- Horizontal spread = magnitude of effect.
- Colors = feature value (e.g., low flowrate = blue, high = red).
- Features at the top: **most important** globally.

### SHAP Interaction Heatmap
- Diagonal = main effects.
- Off-diagonal = **pairwise interaction strength**.
- Brighter = stronger interaction.

### Partial Dependence Plots (PDP)
- 1D: curve of average output vs one feature.
- 2D: surface/contour of two features jointly.
- Flat = no effect; steep or nonlinear = strong effect.
- Twisted/bent 2D surfaces → **synergistic or antagonistic interactions**.

### ANOVA Summary
- **p < 0.05** = statistically significant.
- Shows how much variance in the output each term explains.
- Interactions like `Stabiliser × Flowrate` help detect conditional dependencies.

## License

MIT
