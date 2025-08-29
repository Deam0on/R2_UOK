import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_clean(csv_path, input_cols, output_targets, dropna_required=True, config=None):
    """
    Loads and cleans data from CSV or database depending on config.
    If config['sql_query'] is present, loads from database using SQLAlchemy.
    """
    if config and "sql_query" in config and config["sql_query"]:
        from sqlalchemy import create_engine
        db_url = f"postgresql+psycopg2://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"
        engine = create_engine(db_url)
        df = pd.read_sql_query(config["sql_query"], engine)
    else:
        df = pd.read_csv(csv_path)
    if dropna_required:
        df = df.dropna(subset=input_cols + output_targets)
    return df

def build_preprocessor(input_categoricals, input_numerics):
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), input_categoricals),
        ('num', StandardScaler(), input_numerics)
    ])
