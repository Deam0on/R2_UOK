### utils.py (updated)

from tabulate import tabulate
import logging
from  sklearn import metrics
from scipy.stats import skew
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import re


def print_table(df, title=None, floatfmt=".4f"):
    if title:
        print(f"\n{title}")
    print(tabulate(df, headers="keys", floatfmt=floatfmt, tablefmt="pretty"))

def print_summary(title, lines):
    print(f"\n{'='*len(title)}\n{title}\n{'='*len(title)}")
    for line in lines:
        print(f"  - {line}")

def format_feature_name(name):
    return name.replace(":", " × ").replace("_", " ").title()

def print_top_features(shap_values, feature_names, top_n=5):
    mean_shap = shap_values.mean(axis=0)
    top_idx = mean_shap.argsort()[-top_n:][::-1]
    print(f"\nTop {top_n} most impactful features by SHAP:")
    for rank, idx in enumerate(top_idx, 1):
        fname = format_feature_name(feature_names[idx])
        print(f"{rank}. {fname} (mean SHAP = {mean_shap[idx]:.4f})")
    return top_idx

def print_model_metrics(y_true, y_pred):
    print_summary("Model Evaluation", [
        f"R² = {r2_score(y_true, y_pred):.3f}",
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}",
        f"RMSE = {root_mean_squared_error(y_true, y_pred):.3f}"
    ])

def setup_logger(logfile="analysis.log", level=logging.INFO):
    logging.basicConfig(
        filename=logfile,
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_imbalance(df, config):
    print_summary("Checking for feature imbalances", [])
    for col in config["input_numerics"]:
        s = skew(df[col].dropna())
        if abs(s) > 1:
            print(f"⚠️  {col} is highly skewed (skew = {s:.2f})")
    for target in config["output_targets"]:
        s = skew(df[target].dropna())
        if abs(s) > 1:
            print(f"⚠️  Output {target} is highly skewed (skew = {s:.2f})")



def clean_anova_terms(index_list):
    cleaned = []
    for i in index_list:
        if "Intercept" in i:
            cleaned.append("Intercept")
            continue

        # Replace Q("X") with just X
        i = re.sub(r'Q\\("([^"]+)"\\)', r'\1', i)

        # Replace C(Q("X")) with just X (if it slipped in)
        i = re.sub(r'C\\("([^"]+)"\\)', r'\1', i)

        # Replace [T.Value] or [Value] with = Value
        i = re.sub(r'\\[T\\.(.*?)\\]', r'= \1', i)
        i = re.sub(r'\\[(.*?)\\]', r'= \1', i)

        # Replace colons (:) with × to indicate interaction
        i = i.replace(":", " × ")

        cleaned.append(i)

    return cleaned


