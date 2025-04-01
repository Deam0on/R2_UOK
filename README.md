# Trend Analysis & Feature Importance Tool

A modular, CLI-compatible Python package for exploratory data analysis, regression modeling, feature importance ranking, and SHAP interpretation.

---

## Features

- Correlation Matrix and PCA
- OLS/WLS Regression with p-value filtering
- Random Forests with Cross-Validation
- SHAP Value Visualization and Feature Ranking
- Partial Dependence Plots (1D & 2D)
- ANOVA with Effect Sizes
- YAML config support
- CLI Interface for pipeline control
- Pip-installable as a Python package

---

## Installation

### Install from GitHub
```bash
pip install git+https://github.com/yourusername/trend_analysis.git
```

### Or install locally
```bash
git clone https://github.com/yourusername/trend_analysis.git
cd trend_analysis
pip install .
```

---

## CLI Usage

### Using command-line arguments
```bash
trend-analysis \
  --csv /path/to/data.csv \
  --categoricals "Category 1" "Category 2" \
  --numerics "Numeric 1" "Numeric 2" \
  --targets " Output 1" " Output 2"
```

### Using a YAML config
```bash
trend-analysis --config config.yaml
```

---

## YAML Configuration Example

```yaml
csv_path: /content/data/asmodeus.csv
input_categoricals:
  - Category 1
  - Category 2
input_numerics:
  - Numeric 1
  - Numeric 2
output_targets:
  - Output 1
  - Output 2
dropna_required: true
use_wls: true
significant_only: true
```

---

## Development

### Project structure
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
└── README.md
```

---

## Output

- Terminal-based summaries of model fits, cross-validation, and top features
- Interactive plots (correlation heatmaps, PCA, SHAP, PDP)
- Regression and ANOVA summaries with filtering
- Highlighted feature interactions and residual analysis

---

## License

MIT License

---

## ontributions

Feel free to open issues or submit pull requests for improvements!
