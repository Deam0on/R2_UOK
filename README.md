# Trend Analysis and Feature Importance Tool

This is a modular, CLI-compatible Python package for trend analysis, regression modeling, feature importance evaluation, SHAP interpretability, and statistical analysis.

## Features

- Correlation matrix heatmap and PCA
- OLS/WLS regression with significance filtering
- Random Forest regression with cross-validation
- SHAP value analysis and ranking
- SHAP interaction heatmaps
- Partial dependence plots (1D and 2D)
- ANOVA and 3-way interaction modeling
- CLI interface for scripted use
- YAML configuration file support

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
trend-analysis --csv path/to/data.csv --categoricals API --numerics Flowrate --targets "Mean size"
```

### Full example

```bash
trend-analysis \
  --csv path/to/data.csv \
  --categoricals API Stabilizers "Mixing chambers" \
  --numerics Flowrate \
  --targets "Mean num nm" "Mean vol nm" \
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
csv_path: /content/data/asmodeus.csv
input_categoricals:
  - API
  - Stabilizers
  - Mixing chambers
input_numerics:
  - Flowrate
output_targets:
  - Mean num nm
  - Mean vol nm
dropna_required: true
use_wls: true
significant_only: true
save_plots: true
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

## Programmatic Usage

You can also import and run the analysis from Python:

```python
from trend_analysis.config import config
from trend_analysis.main import main

config["csv_path"] = "/path/to/data.csv"
main(config)
```

## Output

- Printed summaries of regression and ANOVA
- Cross-validated R-squared scores
- SHAP summaries and feature importances
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
