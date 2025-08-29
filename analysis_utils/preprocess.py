"""
Preprocessing utilities for data cleaning and feature engineering.
"""

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
    """
    if config and "sql_query" in config and config["sql_query"]:
        

        db_url = f"postgresql+psycopg2://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"
        engine = create_engine(db_url)
        df = pd.read_sql_query(config["sql_query"], engine)
    else:
        df = pd.read_csv(csv_path)
    if dropna_required:
        df = df.dropna(subset=input_cols + output_targets)
    return df


def build_preprocessor(input_categoricals, input_numerics):
    """
    Builds a ColumnTransformer for preprocessing input features.

    Args:
        input_categoricals (list): List of categorical input feature names.
        input_numerics (list): List of numeric input feature names.

    Returns:
        ColumnTransformer: A ColumnTransformer object configured with OneHotEncoder for categorical features
        and StandardScaler for numeric features.
    """
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                input_categoricals,
            ),
            ("num", StandardScaler(), input_numerics),
        ]
    )
