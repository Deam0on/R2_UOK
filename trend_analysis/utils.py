### utils.py (updated)

from tabulate import tabulate
import logging
import sklearn.metrics
from scipy.stats import skew


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
    from sklearn.metrics import r2_score, mean_absolute_error
    print_summary("Model Evaluation", [
        f"R² = {r2_score(y_true, y_pred):.3f}",
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}",
        f"RMSE = {sklearn.metrics.root_mean_squared_error(y_true, y_pred):.3f}"
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
