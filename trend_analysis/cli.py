### cli.py (modified)

import argparse
import yaml
from trend_analysis.config import config as default_config
from trend_analysis.main import main as run_main

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Trend & Feature Importance Analyzer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--csv", type=str, help="Path to the input CSV file")
    parser.add_argument("--categoricals", nargs="+", help="List of categorical input columns")
    parser.add_argument("--numerics", nargs="+", help="List of numeric input columns")
    parser.add_argument("--targets", nargs="+", help="List of target output columns")
    parser.add_argument("--no-dropna", action="store_true", help="Do not drop rows with missing values")
    parser.add_argument("--ols", action="store_true", help="Use ordinary least squares instead of WLS")
    parser.add_argument("--all", action="store_true", help="Show all results, not only significant ones")
    parser.add_argument("--save-plots", action="store_true", help="Save plots instead of displaying them")
    parser.add_argument("--run-rf", action="store_true", help="Run Random Forest")
    parser.add_argument("--run-shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--run-anova", action="store_true", help="Run ANOVA")
    parser.add_argument("--run-pdp", action="store_true", help="Run PDP plots")
    parser.add_argument("--run-eval", action="store_true", help="Print evaluation metrics")
    parser.add_argument("--run-cv", action="store_true", help="Enable cross-validation")
    parser.add_argument("--run-imbalance-check", action="store_true", help="Check data imbalance")
    parser.add_argument("--generate-summary", action="store_true", help="Generate a human-readable summary at the end")


    return parser.parse_args()

def main():
    args = parse_args()
    config = default_config.copy()

    if args.config:
        config.update(load_yaml_config(args.config))
    if args.csv:
        config["csv_path"] = args.csv
    if args.categoricals:
        config["input_categoricals"] = args.categoricals
    if args.numerics:
        config["input_numerics"] = args.numerics
    if args.targets:
        config["output_targets"] = args.targets
    if args.no_dropna:
        config["dropna_required"] = False
    if args.ols:
        config["use_wls"] = False
    if args.all:
        config["significant_only"] = False

    config["save_plots"] = args.save_plots
    config["run_rf"] = args.run_rf
    config["run_shap"] = args.run_shap
    config["run_anova"] = args.run_anova
    config["run_pdp"] = args.run_pdp
    config["run_eval"] = args.run_eval
    config["run_cv"] = args.run_cv
    config["run_imbalance_check"] = args.run_imbalance_check
    config["generate_summary"] = args.generate_summary

    run_main(config)

if __name__ == "__main__":
    main()