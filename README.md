# R²UOK: Resampling & Regressing Under Ominous Knowledge

A comprehensive, production-ready Python package for advanced statistical analysis, machine learning modeling, and interpretability analysis. R²UOK provides a complete pipeline for data analysis with robust error handling, organized output structure, and enterprise-grade logging.

**"You're doing cross-validation and already know the model won't generalize."**

## Overview

R²UOK is a modular, CLI-compatible Python package designed for comprehensive trend analysis, regression modeling, feature importance evaluation, SHAP interpretability, and statistical analysis. The package emphasizes reproducibility, maintainability, and ease of use while providing enterprise-grade features such as structured logging, organized output management, and comprehensive error handling.OK: Resampling & Regressing Under Ominous Knowledge

You're doing cross-validation and already know the model won’t generalize.

A modular, CLI-compatible Python package for trend analysis, regression modeling, feature importance evaluation, SHAP interpretability, and statistical analysis.

## Key Features

### Core Analysis Capabilities
- **Data Preprocessing**: Advanced data cleaning, missing value handling, and categorical variable processing with reference level optimization
- **Exploratory Data Analysis**: Correlation matrix heatmaps, PCA analysis with variance explanation, and comprehensive data profiling
- **Regression Modeling**: OLS/WLS regression with automatic significance filtering and robust diagnostics
- **Machine Learning**: Random Forest regression with cross-validation, hyperparameter tuning, and comprehensive evaluation metrics
- **Feature Engineering**: Automated categorical encoding, numerical scaling, and feature transformation pipeline

### Advanced Interpretability
- **SHAP Analysis**: Complete SHAP value computation with feature ranking, interaction analysis, and comprehensive visualizations
- **Partial Dependence Analysis**: 1D and 2D partial dependence plots with condensed summary analysis to avoid plot overload
- **Permutation Importance**: Model-agnostic feature importance with statistical significance testing
- **ANOVA Analysis**: Multi-depth interaction support with hierarchical model comparison and effect size quantification

### Production-Ready Infrastructure
- **Organized Output Management**: Timestamped result directories with structured subfolder organization (ANOVA/, PDP/, SHAP/, Models/, Visualizations/, Logs/)
- **Comprehensive Logging**: Multi-level logging with detailed error tracking, performance monitoring, and analysis summaries
- **Configuration Management**: YAML-based configuration with validation, templating, and environment-specific settings
- **CLI Interface**: Full command-line interface with argument validation, help documentation, and batch processing support
- **Error Handling**: Robust try-catch blocks throughout with graceful failure recovery and detailed error reporting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for installation from GitHub)

### Install from GitHub

```bash
git clone https://github.com/Deam0on/R2_UOK.git
cd R2_UOK
pip install -r requirements.txt
```

### Install Dependencies Only

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap statsmodels patsy pyyaml tabulate psycopg2-binary sqlalchemy
```

## Quick Start

### Using Configuration File (Recommended)

1. Copy the configuration template:
```bash
cp config/config_placeholder.yaml config/config.yaml
```

2. Edit `config/config.yaml` with your data settings:
```yaml
# Database connection (or use csv_path for file input)
db_connection:
  host: "your-database-host"
  database: "your-database-name"
  username: "your-username"
  password: "your-password"

# Analysis configuration
input_categoricals: ["category1", "category2"]
input_numerics: ["numeric1", "numeric2"] 
output_targets: ["target1", "target2"]

# Analysis options
run_rf: true
run_shap: true
run_pdp: true
run_anova: true
save_plots: true
```

3. Run the complete analysis pipeline:
```bash
python -m functions.cli --config config/config.yaml
```

### Using CSV File Input

For CSV file input instead of database:

```bash
python -m functions.cli \
  --csv path/to/your/data.csv \
  --categoricals Category1 Category2 Category3 \
  --numerics Numeric1 Numeric2 \
  --targets Target1 Target2 \
  --save-plots \
  --run-rf --run-shap --run-pdp --run-anova
```

## Configuration Reference

### Complete YAML Configuration Example

```yaml
# Data Source Configuration (choose one)
# Option 1: Database connection
db_connection:
  host: "localhost"
  port: 5432
  database: "research_db"
  username: "analyst"
  password: "secure_password"
  
# Option 2: CSV file input
# csv_path: "/path/to/your/data.csv"

# SQL query for database input
sql_query: "SELECT * FROM your_table WHERE condition = 'value'"

# Column Specifications
input_categoricals:
  - lnp_assembly
  - lr_cargo
  - ll_cargo
  - pd_buffer
  
input_numerics:
  - np_ratio
  - temperature
  - ph_level
  
output_targets:
  - diameter_nm
  - pdindex
  - efficiency

# Data Processing Options
dropna_required: true
min_group_size: 10

# Reference Level Settings (optional)
reference_levels:
  lnp_assembly: "False"
  lr_cargo: "Other"
  
# Analysis Configuration
run_rf: true          # Random Forest modeling
run_shap: true        # SHAP analysis
run_pdp: true         # Partial Dependence Plots
run_anova: true       # ANOVA analysis
save_plots: true      # Save visualizations

# Model Parameters
rf_params:
  n_estimators: 100
  max_depth: 10
  random_state: 42
  
anova_params:
  max_interaction_depth: 8
```

### CLI Parameters Reference

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--config` | str | No | Path to YAML configuration file (recommended approach) |
| `--csv` | str | Conditional | Path to CSV file (required if no database config) |
| `--categoricals` | list | Conditional | Categorical input columns (required if no config file) |
| `--numerics` | list | Conditional | Numeric input columns (required if no config file) |
| `--targets` | list | Conditional | Target output variables (required if no config file) |
| `--save-plots` | flag | No | Save plots to organized output folders |
| `--run-rf` | flag | No | Enable Random Forest analysis |
| `--run-shap` | flag | No | Enable SHAP interpretability analysis |
| `--run-pdp` | flag | No | Enable Partial Dependence Plot analysis |
| `--run-anova` | flag | No | Enable ANOVA statistical analysis |
| `--no-dropna` | flag | No | Keep rows with missing values in required columns |
| `--output-dir` | str | No | Custom output directory (default: output/) |


## Output Structure

R²UOK organizes all analysis results into timestamped directories with structured subfolders for easy navigation and analysis:

```
output/
└── results_20250829_135709/
    ├── analysis_results.txt          # High-level summary
    ├── data_head.csv                 # Data sample for verification
    ├── X_target1.csv                 # Feature matrix for target1
    ├── X_target2.csv                 # Feature matrix for target2
    ├── ANOVA/                        # Statistical analysis results
    │   ├── anova_target1_depth_1.csv
    │   ├── anova_target1_depth_2.csv
    │   ├── anova_target2_depth_1.csv
    │   └── ...
    ├── Models/                       # Trained models and metrics
    │   ├── rf_model_target1.pkl
    │   ├── rf_model_target2.pkl
    │   ├── rf_metrics_target1.csv
    │   └── rf_metrics_target2.csv
    ├── PDP/                         # Partial dependence analysis
    │   ├── pdp_feature1_target1.csv
    │   ├── pdp_feature2_target1.csv
    │   ├── pdp_feature1_feature2_target1.csv  # 2D PDPs
    │   └── ...
    ├── SHAP/                        # SHAP interpretability results
    │   ├── shap_values_target1.csv
    │   ├── shap_feature_importance_target1.csv
    │   ├── shap_interactions_target1.csv
    │   └── ...
    ├── Visualizations/              # All generated plots
    │   ├── correlation_matrix.png
    │   ├── pca_analysis.png
    │   ├── shap_summary_target1.png
    │   └── ...
    └── Logs/                        # Execution logs
        ├── analysis.log             # Main analysis log
        └── summary.log              # Summary generation log
```

### Key Output Features

- **Timestamped Organization**: Each analysis run creates a unique timestamped folder
- **Structured Subfolders**: Results organized by analysis type for easy navigation
- **Machine-Readable Formats**: All numerical results saved as CSV for downstream analysis
- **Comprehensive Logging**: Detailed execution logs with error tracking and performance metrics
- **No Overwriting**: Previous results are always preserved

## Advanced Usage

### Programmatic Access

```python
import yaml
import os
from functions.main import main
from config.config import load_config, validate_config

# Load and validate configuration
config_path = "config/config.yaml"
config = load_config(config_path)
validated_config = validate_config(config)

# Run complete analysis pipeline
try:
    main(validated_config)
    print("Analysis completed successfully!")
except Exception as e:
    print(f"Analysis failed: {e}")
```

### Batch Processing Multiple Datasets

```python
import glob
from functions.main import main
from config.config import load_config

# Process multiple configuration files
config_files = glob.glob("configs/*.yaml")
for config_file in config_files:
    print(f"Processing {config_file}...")
    config = load_config(config_file)
    main(config)
```

### Custom Analysis with Specific Components

```python
from analysis_utils.modeling import run_random_forest
from analysis_utils.shap_analysis import run_shap_analysis
from analysis_utils.anova import run_anova_analysis

# Load your data
import pandas as pd
data = pd.read_csv("your_data.csv")

# Run specific analysis components
rf_results = run_random_forest(data, target_col="your_target")
shap_results = run_shap_analysis(rf_results['model'], data)
anova_results = run_anova_analysis(data, "your_target")
```



## Summary Analysis

R²UOK includes a powerful summary analysis tool that provides condensed insights across all analysis types:

### Condensed PDP Analysis

The summary script addresses the common issue of PDP analysis generating hundreds of individual plots by providing:

- **Effect Magnitude Ranking**: Features ranked by their partial dependence effect size
- **Parameter-Level Grouping**: Results aggregated by parameter categories
- **Top Feature Detailed Plots**: Focused visualization on the most impactful features
- **Multi-Panel Summaries**: Consolidated view of key relationships

```bash
# Run summary analysis on latest results
python -m functions.summary

# Run summary on specific results folder
python -m functions.summary output/results_20250829_135709
```

### Summary Output Features

- **Cross-Analysis Integration**: Combines SHAP, ANOVA, and PDP results
- **Feature Importance Synthesis**: Unified ranking across different importance methods
- **Parameter-Level Insights**: Groups individual features by their underlying parameters
- **Statistical Significance**: Highlights statistically significant relationships
- **Condensed Visualizations**: Reduces hundreds of plots to essential insights

## Project Architecture

```
R2_UOK/                              # Root directory
├── analysis_utils/                  # Core analysis modules
│   ├── __init__.py                  # Package initialization
│   ├── anova.py                     # ANOVA statistical analysis
│   ├── modeling.py                  # Machine learning models
│   ├── pdp_analysis.py              # Partial dependence plots  
│   ├── preprocess.py                # Data preprocessing pipeline
│   ├── shap_analysis.py             # SHAP interpretability
│   ├── utils.py                     # Utility functions
│   └── __pycache__/                 # Compiled Python files
├── config/                          # Configuration management
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration validation
│   ├── config.yaml                  # Main configuration file
│   ├── config_placeholder.yaml      # Template configuration
│   └── __pycache__/                 # Compiled Python files
├── functions/                       # Main execution modules
│   ├── __init__.py                  # Package initialization
│   ├── cli.py                       # Command-line interface
│   ├── main.py                      # Main analysis pipeline
│   ├── plotter.py                   # Visualization utilities
│   ├── summary.py                   # Analysis summarization
│   ├── visualization.py             # Advanced plotting
│   └── __pycache__/                 # Compiled Python files
├── output/                          # Analysis results
│   ├── __init__.py                  # Package initialization
│   └── results_YYYYMMDD_HHMMSS/     # Timestamped result folders
│       ├── ANOVA/                   # Statistical analysis results
│       ├── Models/                  # Trained models
│       ├── PDP/                     # Partial dependence plots
│       ├── SHAP/                    # SHAP analysis results
│       ├── Visualizations/          # Generated plots
│       └── Logs/                    # Execution logs
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # License file
└── __init__.py                      # Root package initialization
```

## Dependencies

### Core Requirements

```txt
pandas>=1.3.0                       # Data manipulation and analysis
numpy>=1.21.0                       # Numerical computing
scikit-learn>=1.0.0                 # Machine learning algorithms
matplotlib>=3.5.0                   # Plotting and visualization
seaborn>=0.11.0                     # Statistical data visualization
shap>=0.40.0                        # SHAP interpretability
statsmodels>=0.13.0                 # Statistical analysis
patsy>=0.5.0                        # Statistical formulas
pyyaml>=6.0                         # YAML configuration parsing
tabulate>=0.9.0                     # Table formatting
psycopg2-binary>=2.9.0              # PostgreSQL database adapter
sqlalchemy>=1.4.0                   # SQL toolkit and ORM
```

### Optional Dependencies

```txt
jupyter>=1.0.0                      # Notebook environment
plotly>=5.0.0                       # Interactive plotting
dash>=2.0.0                         # Web applications
streamlit>=1.0.0                    # Data apps
```

## Troubleshooting

### Common Issues and Solutions

#### Database Connection Issues
```bash
# Error: Could not connect to database
# Solution: Verify connection parameters in config.yaml
db_connection:
  host: "correct-host-address"
  port: 5432  # Ensure correct port
  database: "database-name"
  username: "valid-username" 
  password: "correct-password"
```

#### Memory Issues with Large Datasets
```python
# For large datasets, consider data sampling
# Add to your config.yaml:
sample_size: 10000  # Limit to 10k rows
random_state: 42    # For reproducible sampling
```

#### SHAP Analysis Performance
```yaml
# Reduce SHAP computation time for large datasets
shap_params:
  max_display: 20      # Show top 20 features only
  sample_size: 1000    # Use sample for SHAP calculation
```

#### PDP Analysis Overload
```bash
# Use summary analysis to avoid hundreds of PDP plots
python -m functions.summary  # Generates condensed PDP insights
```

### Performance Optimization

#### For Large Datasets
- Use database sampling: `LIMIT` clause in SQL queries
- Enable parallel processing where available
- Consider feature selection before analysis
- Use condensed analysis options in summary script

#### For Production Environments
- Set up dedicated output directories
- Configure appropriate logging levels
- Use configuration templates for consistent analysis
- Implement automated result archival

## Best Practices

### Configuration Management
- Use separate config files for different projects
- Version control your configuration files
- Use descriptive names for analysis parameters
- Document custom reference levels and preprocessing steps

### Data Preparation
- Ensure consistent data types across runs
- Handle missing values appropriately for your domain
- Validate categorical variable levels
- Check for data distribution changes between analyses

### Result Interpretation
- Always review the analysis logs for warnings
- Compare results across different model types
- Use summary analysis for high-level insights
- Validate important findings with domain experts

### Code Maintenance
- Follow the established project structure
- Add comprehensive docstrings for new functions
- Use consistent error handling patterns
- Test configurations before production runs

## Contributing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Deam0on/R2_UOK.git
cd R2_UOK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black isort mypy  # Development tools
```

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write comprehensive docstrings for all functions
- Include error handling with informative messages
- Add logging statements for debugging and monitoring

### Testing
- Write unit tests for new functionality
- Test with various data configurations
- Validate output formats and structures
- Test error handling and edge cases

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support and Contact

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/Deam0on/R2_UOK/issues)
- **Documentation**: Additional documentation available in individual module docstrings
- **Examples**: Check the `config/` directory for configuration examples

## Acknowledgments

R²UOK builds upon the excellent work of the scientific Python community, particularly:
- scikit-learn for machine learning algorithms
- SHAP for model interpretability
- statsmodels for statistical analysis
- matplotlib and seaborn for visualization
- pandas for data manipulation

## Changelog

### Version 2.0.0 (Current)
- Added organized output directory structure
- Implemented condensed PDP analysis in summary script
- Enhanced error handling and logging throughout
- Added comprehensive configuration validation
- Improved CLI interface with better argument handling
- Added support for both database and CSV input

### Version 1.0.0
- Initial release with core analysis capabilities
- Basic CLI interface and configuration support
- Core SHAP, ANOVA, and Random Forest analysis
