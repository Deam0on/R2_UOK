### main.py (modified)

from trend_analysis.config import config as default_config
from trend_analysis.utils import (
    print_table, print_summary, format_feature_name,
    print_top_features, print_model_metrics, setup_logger, check_imbalance
)
from trend_analysis.preprocess import load_and_clean, build_preprocessor
from trend_analysis.visualization import show_correlation, plot_pca
from trend_analysis.modeling import fit_linear_model, fit_random_forest
from trend_analysis.shap_analysis import explain_shap
from trend_analysis.pdp_analysis import plot_pdp
from trend_analysis.anova import run_anova
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging
from trend_analysis.utils import clean_anova_terms
from trend_analysis.utils import transform_skewed_columns

def main(config=None):
    setup_logger()
    if config is None:
        config = default_config

    df = load_and_clean(
        config["csv_path"],
        config["input_categoricals"] + config["input_numerics"],
        config["output_targets"],
        dropna_required=config.get("dropna_required", True)
    )

    for col in config["input_categoricals"]:
        ref = str(config.get("reference_levels", {}).get(col))
        if ref:
            categories = sorted([str(x) for x in df[col].dropna().unique()])
            if ref not in categories:
                print(f"⚠️  Warning: Reference '{ref}' not found in column '{col}'")
            else:
                categories.remove(ref)
                df[col] = pd.Categorical(df[col].astype(str), categories=[ref] + categories, ordered=True)



    if config.get("run_imbalance_check", False):
        skewed_cols = check_imbalance(df, config)
        if skewed_cols:
            print("→ Transforming skewed columns...")
            df = transform_skewed_columns(df, config)

    preprocessor = build_preprocessor(config["input_categoricals"], config["input_numerics"])
    X = preprocessor.fit_transform(df[config["input_categoricals"] + config["input_numerics"]])
    feature_names = preprocessor.get_feature_names_out()

    show_correlation(df, config["input_numerics"], config.get("save_plots", False))
    plot_pca(X, PCA(), config.get("save_plots", False))

    for output in config["output_targets"]:
        y = df[output]

        print_summary(f"Linear Regression for '{output}'", [])
        model = fit_linear_model(X, y, df, output, config["use_wls"], config["significant_only"])
        coef_table = model.summary2().tables[1]
        if config["significant_only"]:
            coef_table = coef_table[coef_table["P>|t|"] < 0.05]
        # print_table(coef_table, title="Significant Coefficients (p < 0.05)")
        coef_table = coef_table.rename(index={"const": "Intercept"})
        print_table(coef_table, title="Significant Coefficients (p < 0.05)")

        if config.get("run_rf", False):
            rf, _ = fit_random_forest(X, y)

            if config.get("run_cv", False):
                scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
                print_summary("Random Forest Cross-Validation", [
                    f"Mean R²: {scores.mean():.3f}",
                    f"Std Dev : {scores.std():.3f}"
                ])

            rf.fit(X, y)
            if config.get("run_eval", False):
                y_pred = rf.predict(X)
                print_model_metrics(y, y_pred)

            if config.get("run_shap", False):
                mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, config["significant_only"], config.get("save_plots", False))
                print_top_features(abs(shap_values), feature_names, top_n=5)

            if config.get("run_pdp", False):
                plot_pdp(rf, X, feature_names, top_idx, config["significant_only"], config.get("save_plots", False))

        if config.get("run_anova", False):
            anova_models = run_anova(df, output, config["input_categoricals"], config["input_numerics"])

            for depth, model in anova_models.items():
                anova_table = model.summary2().tables[1]
                if config["significant_only"]:
                    anova_table = anova_table[anova_table["P>|t|"] < 0.05]

                anova_table.index = clean_anova_terms(anova_table.index)
                print_table(anova_table, title=f"ANOVA: Depth {depth} Significant Terms (p < 0.05)")

