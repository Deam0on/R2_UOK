import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_clean(csv_path, input_cols, output_targets, dropna_required=True):
    df = pd.read_csv(csv_path)
    if dropna_required:
        df = df.dropna(subset=input_cols + output_targets)
    return df

def build_preprocessor(input_categoricals, input_numerics):
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), input_categoricals),
        ('num', StandardScaler(), input_numerics)
    ])
