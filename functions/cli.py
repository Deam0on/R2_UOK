"""
Command-line interface for the trend analysis pipeline.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

from config.config import config as default_config
from functions.main import main as run_main


def setup_cli_logger(output_dir=None):
    """
    Set up logging for the CLI.
    
    Args:
        output_dir (str, optional): Directory to save log file. If None, saves to current directory.
    """
    if output_dir is None:
        log_file = 'cli.log'
    else:
        log_file = os.path.join(output_dir, 'Logs', 'cli.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


def load_yaml_config(path):
    """
    Load a YAML configuration file from the given path.
    
    Args:
        path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary loaded from YAML file.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file contains invalid YAML.
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Loading YAML config from: {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {path}")
        return config
    except FileNotFoundError as e:
        logger.exception(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}") from e
    except yaml.YAMLError as e:
        logger.exception(f"Invalid YAML in config file: {path}")
        raise yaml.YAMLError(f"Invalid YAML in config file: {path}") from e
    except Exception as e:
        logger.exception(f"Unexpected error loading config: {path}")
        raise RuntimeError(f"Failed to load config: {e}") from e


def parse_args():
    """
    Parse command-line arguments for the analysis pipeline.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Trend & Feature Importance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/config.yaml
  %(prog)s --csv data.csv --categoricals Cat1 Cat2 --numerics Num1 Num2 --targets Target1
  %(prog)s --csv data.csv --all --save-plots
        """
    )
    
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
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def create_output_directory():
    """
    Create a timestamped output directory with organized subfolders for results.
    
    Returns:
        str: Path to the created output directory.
        
    Raises:
        OSError: If directory creation fails.
    """
    logger = logging.getLogger(__name__)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"results_{timestamp}")
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create organized subfolders
        subfolders = ['ANOVA', 'PDP', 'SHAP', 'Models', 'Visualizations', 'Logs']
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
            
        logger.info(f"Created output directory with subfolders: {output_dir}")
        return output_dir
    except OSError as e:
        logger.exception(f"Failed to create output directory")
        raise OSError(f"Cannot create output directory: {e}") from e


def cleanup_files_to_output_dir(output_dir):
    """
    Move any remaining output files to the appropriate subfolders in the output directory.
    
    Args:
        output_dir (str): Target output directory.
    """
    logger = logging.getLogger(__name__)
    
    # Files to move to Visualizations folder
    viz_files = [
        'correlation_matrix.png',
        'correlation_matrix.csv', 
        'pca_variance.png'
    ]
    
    # Files to move to Logs folder
    log_files = [
        'summary.log'
    ]
    
    # Move visualization files
    for filename in viz_files:
        if os.path.exists(filename):
            try:
                destination = os.path.join(output_dir, 'Visualizations', filename)
                shutil.move(filename, destination)
                logger.debug(f"Moved {filename} to {destination}")
            except Exception as e:
                logger.warning(f"Failed to move {filename} to output directory: {e}")
    
    # Move log files
    for filename in log_files:
        if os.path.exists(filename):
            try:
                destination = os.path.join(output_dir, 'Logs', filename)
                shutil.move(filename, destination)
                logger.debug(f"Moved {filename} to {destination}")
            except Exception as e:
                logger.warning(f"Failed to move {filename} to output directory: {e}")


def validate_config(config):
    """
    Validate that the configuration has required keys.
    
    Args:
        config (dict): Configuration dictionary to validate.
        
    Raises:
        ValueError: If required configuration keys are missing.
    """
    logger = logging.getLogger(__name__)
    
    required_keys = ["input_categoricals", "input_numerics", "output_targets"]
    missing_keys = [key for key in required_keys if key not in config or not config[key]]
    
    if missing_keys:
        error_msg = f"Missing required configuration keys: {missing_keys}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not config.get("csv_path") and not (config.get("sql_query") and config.get("db_host")):
        error_msg = "Either csv_path or database configuration (sql_query + db_host) must be provided"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug("Configuration validation passed")


def run_summary_script():
    """
    Run the summary script after analysis completion.
    """
    logger = logging.getLogger(__name__)
    
    try:
        summary_path = os.path.join(os.path.dirname(__file__), "summary.py")
        summary_path = os.path.abspath(summary_path)
        
        if os.path.exists(summary_path):
            print("\n[INFO] Summarizing all outputs using summary.py...\n")
            logger.info("Running summary script")
            result = subprocess.run([sys.executable, summary_path], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Summary script failed: {result.stderr}")
                print(f"[WARNING] Summary script failed: {result.stderr}")
            else:
                logger.info("Summary script completed successfully")
        else:
            logger.warning(f"Summary script not found: {summary_path}")
            print(f"[WARNING] Summary script not found: {summary_path}")
            
    except Exception as e:
        logger.exception("Failed to run summary script")
        print(f"[WARNING] Failed to run summary script: {e}")


def main():
    """
    Main entry point for the CLI. Parses arguments, sets up config, creates timestamped output folder, and runs analysis.
    """
    # Initial logger setup (will be reconfigured later with output directory)
    initial_logger = setup_cli_logger()
    
    try:
        args = parse_args()
        
        initial_logger.info("Starting trend analysis CLI")
        initial_logger.debug(f"CLI arguments: {vars(args)}")
        
        # Start with default config
        config = default_config.copy()

        # Create timestamped output directory
        try:
            output_dir = create_output_directory()
            config["output_dir"] = output_dir
            initial_logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            initial_logger.exception("Failed to create output directory")
            print(f"Error: Failed to create output directory: {e}")
            sys.exit(1)

        # Reconfigure logger to use output directory
        logger = setup_cli_logger(output_dir)
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")

        # Load and merge configuration
        if args.config:
            try:
                yaml_config = load_yaml_config(args.config)
                config.update(yaml_config)
            except (FileNotFoundError, yaml.YAMLError, RuntimeError) as e:
                print(f"Error loading config file: {e}")
                sys.exit(1)
        
        # Override with command line arguments
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

        # Set boolean flags
        config["save_plots"] = args.save_plots
        config["run_rf"] = args.run_rf
        config["run_shap"] = args.run_shap
        config["run_anova"] = args.run_anova
        config["run_pdp"] = args.run_pdp
        config["run_eval"] = args.run_eval
        config["run_cv"] = args.run_cv
        config["run_imbalance_check"] = args.run_imbalance_check
        config["generate_summary"] = args.generate_summary

        # Validate configuration
        try:
            validate_config(config)
        except ValueError as e:
            print(f"Configuration error: {e}")
            sys.exit(1)

        logger.info("Configuration validated successfully")
        logger.debug(f"Final config: {config}")

        # Run main analysis
        try:
            run_main(config)
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.exception("Analysis failed")
            print(f"Error: Analysis failed: {e}")
            sys.exit(1)

        # Run summary script
        run_summary_script()
        
        # Clean up any remaining files
        cleanup_files_to_output_dir(output_dir)
        
        logger.info("CLI execution completed successfully")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error in CLI")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
