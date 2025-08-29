import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml

from config.config import config as default_config
from functions.main import main as run_main


def load_yaml_config(path):
    """
    Load a YAML configuration file from the given path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """
    Parse command-line arguments for the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Trend & Feature Importance Analyzer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--csv", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--categoricals", nargs="+", help="List of categorical input columns"
    )
    parser.add_argument("--numerics", nargs="+", help="List of numeric input columns")
    parser.add_argument("--targets", nargs="+", help="List of target output columns")
    parser.add_argument(
        "--no-dropna", action="store_true", help="Do not drop rows with missing values"
    )
    parser.add_argument(
        "--ols", action="store_true", help="Use ordinary least squares instead of WLS"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all results, not only significant ones"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots instead of displaying them",
    )
    parser.add_argument("--run-rf", action="store_true", help="Run Random Forest")
    parser.add_argument("--run-shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--run-anova", action="store_true", help="Run ANOVA")
    parser.add_argument("--run-pdp", action="store_true", help="Run PDP plots")
    parser.add_argument(
        "--run-eval", action="store_true", help="Print evaluation metrics"
    )
    parser.add_argument("--run-cv", action="store_true", help="Enable cross-validation")
    parser.add_argument(
        "--run-imbalance-check", action="store_true", help="Check data imbalance"
    )
    parser.add_argument(
        "--generate-summary",
        action="store_true",
        help="Generate a human-readable summary at the end",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the CLI. Parses arguments, sets up config, creates timestamped output folder, and runs analysis.
    """
    args = parse_args()
    config = default_config.copy()

    # Create a timestamped output folder for new results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

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

    # Automatically run the summary after analysis
    summary_path = os.path.join(os.path.dirname(__file__), "summary.py")
    summary_path = os.path.abspath(summary_path)
    if os.path.exists(summary_path):
        print("\n[INFO] Summarizing all outputs using summary.py...\n")
        subprocess.run([sys.executable, summary_path])


if __name__ == "__main__":
    main()
