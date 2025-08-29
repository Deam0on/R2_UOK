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
python functions/cli.py --csv path/to/data.csv --categoricals Category1 --numerics Numeric1 --targets Target1
```

### Full example

```bash
python functions/cli.py \
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
python -m functions/cli.py --config config/config.yaml
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

You can also run the analysis from Python by loading your YAML config and calling the main function:

```python
import yaml
from trend_analysis.main import main

with open("trend_analysis/config.yaml", "r") as f:
  config = yaml.safe_load(f)
main(config)
```



## Output

- All analysis results (regression, ANOVA, SHAP, PDP, etc.) are saved as CSV or TXT files in a timestamped subfolder under the `output/` directory (e.g., `output/results_YYYYMMDD_HHMMSS/`).
- No results are printed to the terminal or saved as images by default; all numeric results are machine-readable.
- Plots (e.g., feature importances, PDPs, correlation matrices) can be generated from the CSV outputs using the provided `plotter.py` script.
- Robust error handling: if analysis for one output target fails, the pipeline continues for the rest, and errors are logged in `analysis.log`.



## Project Structure

```
R2_UOK/
├── analysis_utils/
│   ├── __init__.py
│   ├── anova.py
│   ├── modeling.py
│   ├── pdp_analysis.py
│   ├── preprocess.py
│   ├── shap_analysis.py
│   ├── utils.py
├── config/
│   ├── __init__.py
│   ├── config.py
│   ├── config.yaml
│   ├── config_placeholder.yaml
├── functions/
│   ├── __init__.py
│   ├── cli.py
│   ├── main.py
│   ├── plotter.py
│   ├── summary.py
│   ├── visualization.py
├── output/
│   ├── __init__.py
│   ├── results_YYYYMMDD_HHMMSS/
│   │   ├── analysis_results.txt
│   │   ├── data_head.csv
│   │   ├── ... (all result CSVs)
├── analysis.log
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
```

## Notes

- All new results are saved in timestamped subfolders under `output/` and are never overwritten or backed up.
- The CLI entry point is `functions/cli.py`.
- All results are output as CSV/txt for downstream analysis or plotting.
- See `functions/plotter.py` and `functions/summary.py` for examples of how to visualize or aggregate results from the output files.
- All imports are at the top of each file and all functions have docstrings for clarity and maintainability.

## License

This project is licensed under the MIT License.

## Contributions

Contributions, issues, and feature requests are welcome. Please open a pull request or submit an issue on GitHub.
