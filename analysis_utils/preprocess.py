"""
Preprocessing utilities for data cleaning and feature engineering.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


def load_and_clean(
    csv_path, input_cols, output_targets, dropna_required=True, config=None
):
    """
    Loads and cleans data from CSV or database depending on config.
    If config['sql_query'] is present, loads from database using SQLAlchemy.

    Args:
        csv_path (str): Path to the CSV file.
        input_cols (list): List of input column names.
        output_targets (list): List of output target names.
        dropna_required (bool): Whether to drop rows with NA in input or output columns.
        config (dict, optional): Configuration dictionary, potentially containing SQL query and database connection info.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded and cleaned data.
        
    Raises:
        FileNotFoundError: If CSV file is not found and no SQL query is provided.
        ConnectionError: If database connection fails.
        ValueError: If required columns are missing from the data.
    """
    logger = logging.getLogger(__name__)
    
    try:
        if config and "sql_query" in config and config["sql_query"]:
            logger.info("Loading data from database using SQL query")
            try:
                db_url = f"postgresql+psycopg2://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"
                engine = create_engine(db_url)
                df = pd.read_sql_query(config["sql_query"], engine)
                logger.debug(f"Successfully loaded {len(df)} rows from database")
            except Exception as e:
                logger.exception("Failed to load data from database")
                raise ConnectionError(f"Database connection failed: {e}") from e
        else:
            if not csv_path:
                raise ValueError("No CSV path provided and no SQL query in config")
            logger.info(f"Loading data from CSV file: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                logger.debug(f"Successfully loaded {len(df)} rows from CSV")
            except FileNotFoundError as e:
                logger.exception(f"CSV file not found: {csv_path}")
                raise FileNotFoundError(f"CSV file not found: {csv_path}") from e
            except Exception as e:
                logger.exception(f"Failed to read CSV file: {csv_path}")
                raise ValueError(f"Failed to read CSV file: {e}") from e
        
        # Validate required columns exist
        all_required_cols = input_cols + output_targets
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing from data: {missing_cols}")
            
        logger.debug(f"Data columns: {df.columns.tolist()}")
        logger.debug(f"Data shape before cleaning: {df.shape}")
        
        if dropna_required:
            logger.info("Dropping rows with missing values in required columns")
            initial_rows = len(df)
            df = df.dropna(subset=input_cols + output_targets)
            final_rows = len(df)
            dropped_rows = initial_rows - final_rows
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing values")
            logger.debug(f"Data shape after dropping NA: {df.shape}")
            
        return df
        
    except Exception as e:
        logger.exception("Failed to load and clean data")
        raise


def build_preprocessor(input_categoricals, input_numerics):
    """
    Builds a ColumnTransformer for preprocessing input features.

    Args:
        input_categoricals (list): List of categorical input feature names.
        input_numerics (list): List of numeric input feature names.

    Returns:
        ColumnTransformer: A ColumnTransformer object configured with OneHotEncoder for categorical features
        and StandardScaler for numeric features.
        
    Raises:
        ValueError: If no input features are provided.
    """
    logger = logging.getLogger(__name__)
    
    if not input_categoricals and not input_numerics:
        raise ValueError("No input features provided (both categorical and numeric lists are empty)")
    
    transformers = []
    
    if input_categoricals:
        logger.debug(f"Adding categorical transformer for: {input_categoricals}")
        transformers.append((
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            input_categoricals,
        ))
    
    if input_numerics:
        logger.debug(f"Adding numeric transformer for: {input_numerics}")
        transformers.append((
            "num", 
            StandardScaler(), 
            input_numerics
        ))
    
    try:
        preprocessor = ColumnTransformer(transformers)
        logger.debug("Successfully created ColumnTransformer")
        return preprocessor
    except Exception as e:
        logger.exception("Failed to create ColumnTransformer")
        raise ValueError(f"Failed to create preprocessor: {e}") from e
