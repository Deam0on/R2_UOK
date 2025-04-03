from trend_analysis.config import config as default_config
from trend_analysis.utils import print_table, print_summary, format_feature_name, print_top_features
from trend_analysis.preprocess import load_and_clean, build_preprocessor
from trend_analysis.visualization import show_correlation, plot_pca
from trend_analysis.modeling import fit_linear_model, fit_random_forest
from trend_analysis.shap_analysis import explain_shap
from trend_analysis.pdp_analysis import plot_pdp
from trend_analysis.anova import run_anova
from sklearn.decomposition import PCA
import pandas as pd

def main(config=None):
    if config is None:
        config = default_config

    df = load_and_clean(
        config["csv_path"],
        config["input_categoricals"] + config["input_numerics"],
        config["output_targets"],
        dropna_required=config.get("dropna_required", True)
    )

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
        print_table(coef_table, title="Significant Coefficients (p < 0.05)")

        rf, scores = fit_random_forest(X, y)
        print_summary("Random Forest Cross-Validation", [
            f"Mean R²: {scores.mean():.3f}",
            f"Std Dev : {scores.std():.3f}"
        ])

        mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, config["significant_only"], config.get("save_plots", False))
        print_top_features(abs(shap_values), feature_names, top_n=5)

        plot_pdp(rf, X, feature_names, top_idx, config["significant_only"], config.get("save_plots", False))

        anova_model = run_anova(df, output, config["input_categoricals"], config["input_numerics"])
        anova_table = anova_model.summary2().tables[1]
        if config["significant_only"]:
            anova_table = anova_table[anova_table["P>|t|"] < 0.05]
        print_table(anova_table, title="ANOVA: Significant Terms (p < 0.05)")
