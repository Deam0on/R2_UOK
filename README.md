
# Multi-Parametric Trend Analysis Pipeline

This document describes the data processing and modeling pipeline implemented to analyze the relationship between a set of categorical and numerical input parameters and two quantitative output metrics, with a focus on feature importance, interaction effects, and statistical modeling. The pipeline is implemented in Python and is designed for use with a CSV file formatted with the following columns:

## Data Format

**Inputs:**
- `API` (categorical): The active pharmaceutical ingredient used.
- `Stabiliser` (categorical): The stabilizing agent employed.
- `MixingChamber` (categorical): The mixing chamber design.
- `Flowrate` (numerical, mL/min): The total flowrate during the experiment.

**Outputs:**
- `MeanNumberSize` (numerical, nm): Mean hydrodynamic diameter based on number distribution.
- `MeanNumberSizeStd` (numerical, nm): Standard deviation of number-based size.
- `MeanVolumeSize` (numerical, nm): Mean hydrodynamic diameter based on volume distribution.
- `MeanVolumeSizeStd` (numerical, nm): Standard deviation of volume-based size.

## Methodology

The pipeline performs a sequence of preprocessing steps, modeling tasks, and result interpretation phases as described below.

### 1. Data Preprocessing

- **Handling missing data**: Rows with missing values in any of the required input or output columns are dropped. Optionally, numerical fields (e.g., `Flowrate`) can be imputed using the mean.
- **Categorical encoding**: Categorical variables (`API`, `Stabiliser`, `MixingChamber`) are encoded using One-Hot Encoding (dropping the first level to avoid collinearity).
- **Feature scaling**: Numerical input `Flowrate` is standardized using z-score normalization (mean = 0, std = 1).

### 2. Linear Regression

A linear regression model is trained for each output (MeanNumberSize and MeanVolumeSize) to estimate the contribution of each input parameter to the observed output:

- The resulting coefficients indicate the expected change in output per unit change in the input feature, holding all else constant.
- Positive coefficients imply a direct relationship; negative coefficients imply an inverse relationship.
- This model assumes linearity and independence between predictors.

### 3. Random Forest Regressor

A non-parametric ensemble model (Random Forest) is trained for each output. It handles non-linear relationships and interactions between variables.

- **Feature Importance**: Calculated as the average reduction in impurity (e.g., mean squared error) across all trees when a feature is used for splitting.
- Larger values indicate greater importance in predicting the target variable.

### 4. SHAP (SHapley Additive Explanations)

SHAP values provide a unified measure of feature contribution based on game theory:

- For each sample and feature, the SHAP value quantifies how much that feature contributes to pushing the model output away from the average.
- **SHAP Summary Plot**: Visualizes feature importance, direction of effect, and variability across the dataset.
- Interpretation: Features at the top are most important; color shows feature value (e.g., high or low Flowrate).

### 5. Partial Dependence Plots

Partial dependence plots (PDPs) visualize the marginal effect of one or two features on the predicted output:

- For two features, interaction surfaces can be interpreted to assess non-additive behavior.
- Example: The effect of Flowrate may differ depending on the Stabiliser.

### 6. ANOVA (Analysis of Variance)

A linear model is fitted using ordinary least squares (OLS) with categorical inputs treated explicitly as factors. The model is of the form:

```
output ~ C(API) + C(Stabiliser) + C(MixingChamber) + Flowrate
```

- **R²**: Proportion of variance in the output explained by the model.
- **p-values**: Indicate the significance of each predictor in explaining variance.
- This step quantifies the statistical significance of each categorical factor.

## Interpretation Guidelines

- Use linear regression coefficients to assess additive, linear effects.
- Use Random Forest and SHAP to interpret nonlinear and interaction effects.
- Use PDPs to identify synergistic or antagonistic interactions.
- Use ANOVA to statistically validate categorical effects.
- Cross-compare outputs to identify consistent or divergent trends.

## Output Visualization

Each model produces:
- A barplot of Random Forest feature importances.
- SHAP summary plots per output variable.
- A 2D Partial Dependence surface for the top two interacting features.
- A statistical summary table from ANOVA (including p-values and R²).

## Notes

- Ensure input data is properly cleaned and validated before use.
- Interpretation of SHAP and PDP requires domain expertise.
- The pipeline is extensible for additional outputs or experimental factors.
