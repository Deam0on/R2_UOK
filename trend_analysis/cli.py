import argparse
import yaml
from config import config as default_config
from main import main as run_main

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

    run_main(config)

if __name__ == "__main__":
    main()
