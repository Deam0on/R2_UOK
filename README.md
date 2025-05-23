# R²UOK: Resampling & Regressing Under Ominous Knowledge

You're doing cross-validation and already know the model won’t generalize.

A modular, CLI-compatible Python package for trend analysis, regression modeling, feature importance evaluation, SHAP interpretability, and statistical analysis.

## Features

- Correlation matrix heatmap and PCA
- OLS/WLS regression with significance filtering
- Reference level selection for categorical predictors
- Residual diagnostics and skewness-aware transformations
- Random Forest regression with cross-validation and evaluation
- SHAP value analysis and ranking
- SHAP interaction heatmaps
- Partial dependence plots (1D and 2D)
- ANOVA with multi-depth interaction support
- CLI interface for scripted use
- YAML configuration file support
- Optional saving of plots to PNG

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/yourusername/trend_analysis.git
```

### Install locally from source

```bash
git clone https://github.com/yourusername/trend_analysis.git
cd trend_analysis
pip install .
```

## CLI Usage

### Minimal example

```bash
trend-analysis --csv path/to/data.csv --categoricals Category1 --numerics Numeric1 --targets Target1
```

### Full example

```bash
trend-analysis \
  --csv path/to/data.csv \
  --categoricals Category1 Category2 Category3 \
  --numerics Numeric1 Numeric2 \
  --targets Target1 Target2 \
  --no-dropna \
  --ols \
  --all
```

### Using a YAML configuration

```bash
trend-analysis --config config.yaml
```

## YAML Config Example

```yaml
csv_path: /path/to/data.csv
input_categoricals:
  - Category1
  - Category2
  - Category3
input_numerics:
  - Numeric1
  - Numeric2
output_targets:
  - Target1
  - Target2
dropna_required: true
use_wls: true
significant_only: true
save_plots: true
run_rf: true
run_cv: true
run_eval: true
run_shap: true
run_pdp: false
run_anova: true
run_imbalance_check: true
reference_levels:
  Category1: SomeReference
  Category2: OtherReference
```

## Available CLI Parameters

| Parameter            | Required | Description                                                        |
|----------------------|----------|--------------------------------------------------------------------|
| `--csv`              | Yes      | Path to the input CSV file                                         |
| `--categoricals`     | Yes      | List of categorical input columns                                  |
| `--numerics`         | Yes      | List of numeric input columns                                      |
| `--targets`          | Yes      | List of target output variables                                    |
| `--config`           | No       | Path to YAML configuration file                                    |
| `--no-dropna`        | No       | Keep rows with missing values                                      |
| `--ols`              | No       | Use Ordinary Least Squares instead of Weighted Least Squares       |
| `--all`              | No       | Show all results, not just significant ones                        |
| `--save-plots`       | No       | Save figures to PNG instead of showing them                        |

## Programmatic Usage

You can also import and run the analysis from Python:

```python
from trend_analysis.config import config
from trend_analysis.main import main

config["csv_path"] = "/path/to/data.csv"
config["input_categoricals"] = ["Category1", "Category2"]
config["input_numerics"] = ["Numeric1", "Numeric2"]
config["output_targets"] = ["Target1", "Target2"]
main(config)
```

## Output

- Printed summaries of regression and ANOVA
- Cross-validated R-squared scores and error metrics
- SHAP summaries and feature importances
- Automatically transformed skewed variables
- Plots (interactive or saved as PNG)
- Top interactions and residual diagnostics

## Project Structure

```
trend_analysis/
├── trend_analysis/
│   ├── __init__.py
│   ├── main.py
│   ├── cli.py
│   ├── config.py
│   ├── preprocess.py
│   ├── visualization.py
│   ├── modeling.py
│   ├── shap_analysis.py
│   ├── pdp_analysis.py
│   ├── anova.py
│   └── config.yaml
├── setup.py
├── README.md
```

## License

This project is licensed under the MIT License.

## Contributions

Contributions, issues, and feature requests are welcome. Please open a pull request or submit an issue on GitHub.
