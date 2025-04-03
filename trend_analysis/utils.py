from tabulate import tabulate

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
