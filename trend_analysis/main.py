### main.py (modified)

from trend_analysis.config import config as default_config
from trend_analysis.utils import (
    print_table, print_summary, format_feature_name,
    print_top_features, print_model_metrics, setup_logger, check_imbalance,
    transform_skewed_columns, clean_linear_terms, clean_anova_terms, generate_auto_summary
)
from trend_analysis.preprocess import load_and_clean, build_preprocessor
from trend_analysis.visualization import show_correlation, plot_pca
from trend_analysis.modeling import fit_linear_model, fit_random_forest
from trend_analysis.shap_analysis import explain_shap
from trend_analysis.pdp_analysis import plot_pdp
from trend_analysis.anova import run_anova
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import pandas as pd
import logging
from sklearn.metrics import root_mean_squared_error
import numpy as np

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

    if not config.get("dropna_required", False):
        imputer = SimpleImputer(strategy="mean")
        df[config["input_numerics"]] = imputer.fit_transform(df[config["input_numerics"]])

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
            print(f"→ Transforming skewed columns: {', '.join(skewed_cols)}")
            df = transform_skewed_columns(df, config)

    preprocessor = build_preprocessor(config["input_categoricals"], config["input_numerics"])
    X = preprocessor.fit_transform(df[config["input_categoricals"] + config["input_numerics"]])
    feature_names = preprocessor.get_feature_names_out()

    show_correlation(df, config["input_numerics"], config.get("save_plots", False))
    plot_pca(X, PCA(), config.get("save_plots", False))

    target_summaries = {}

    for output in config["output_targets"]:
        y = df[output]
        target_summary = {}

        print_summary(f"Linear Regression for '{output}'", [])
        model = fit_linear_model(X, y, df, output, config["use_wls"], config["significant_only"])

        coef_table = model.summary2().tables[1]
        new_index = ["Intercept"] + list(feature_names)
        coef_table.index = new_index[: len(coef_table)]
        coef_table.index = clean_linear_terms(coef_table.index)

        ref_terms = [f"{col} = {config['reference_levels'][col]}" for col in config["input_categoricals"] if col in config.get("reference_levels", {})]
        if ref_terms:
            if "Intercept" in coef_table.index:
                coef_table.rename(index={"Intercept": f"Intercept ({', '.join(ref_terms)})"}, inplace=True)

        if config["significant_only"]:
            coef_table = coef_table[coef_table["P>|t|"] < 0.05]

        print_table(coef_table, title="Significant Coefficients (p < 0.05)")
        target_summary["coef_table"] = coef_table

        if config.get("run_rf", False):
            rf, _ = fit_random_forest(X, y)
            scores = None

            if config.get("run_cv", False):
                scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
                print_summary("Random Forest Cross-Validation", [
                    f"Mean R²: {scores.mean():.3f}",
                    f"Std Dev : {scores.std():.3f}"
                ])
                target_summary["cv_scores"] = scores

            rf.fit(X, y)
            if config.get("run_eval", False):
                y_pred = rf.predict(X)
                print_model_metrics(y, y_pred)
                target_summary["metrics"] = {
                    "r2": rf.score(X, y),
                    "mae": np.mean(np.abs(y - y_pred)),
                    "rmse": root_mean_squared_error(y, y_pred, squared=False)
                }

            if config.get("run_shap", False):
                mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, config["significant_only"], config.get("save_plots", False))
                print_top_features(abs(shap_values), feature_names, top_n=5)
                shap_top = []
                mean_abs = np.abs(shap_values).mean(axis=0)
                for idx in top_idx:
                    dir = "positive" if np.corrcoef(X[:, idx], y)[0, 1] > 0 else "negative"
                    shap_top.append((feature_names[idx], mean_abs[idx], dir))
                target_summary["shap_top"] = shap_top

        if config.get("run_anova", False):
            anova_models = run_anova(df, output, config["input_categoricals"], config["input_numerics"])
            all_anova = []
            for depth, model in anova_models.items():
                anova_table = model.summary2().tables[1]
                if config["significant_only"]:
                    anova_table = anova_table[anova_table["P>|t|"] < 0.05]
                anova_table.index = clean_anova_terms(anova_table.index)
                if ref_terms:
                    if "Intercept" in anova_table.index:
                        anova_table.rename(index={"Intercept": f"Intercept ({', '.join(ref_terms)})"}, inplace=True)
                print_table(anova_table, title=f"ANOVA: Depth {depth} Significant Terms (p < 0.05)")
                if depth == 1:
                    target_summary["anova_table"] = anova_table

        target_summaries[output] = target_summary

    if config.get("generate_summary", False):
        generate_auto_summary(target_summaries, config)
