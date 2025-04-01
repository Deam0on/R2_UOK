from config import config
from preprocess import load_and_clean, build_preprocessor
from visualization import show_correlation, plot_pca
from modeling import fit_linear_model, fit_random_forest
from shap_analysis import explain_shap
from pdp_analysis import plot_pdp
from anova import run_anova
from sklearn.decomposition import PCA

def main():
    df = load_and_clean(
        config["csv_path"],
        config["input_categoricals"] + config["input_numerics"],
        config["output_targets"],
        dropna_required=config["dropna_required"]
    )

    preprocessor = build_preprocessor(config["input_categoricals"], config["input_numerics"])
    X = preprocessor.fit_transform(df[config["input_categoricals"] + config["input_numerics"]])
    feature_names = preprocessor.get_feature_names_out()

    show_correlation(df, config["input_numerics"])
    plot_pca(X, PCA())

    for output in config["output_targets"]:
        y = df[output]

        model = fit_linear_model(X, y, df, output, config["use_wls"], config["significant_only"])
        print(model.summary2().tables[1][model.pvalues < 0.05] if config["significant_only"] else model.summary())

        rf, scores = fit_random_forest(X, y)
        print(f"Cross-validated R²: {scores.mean():.3f} ± {scores.std():.3f}")

        mean_shap, top_idx, shap_values = explain_shap(rf, X, feature_names, config["significant_only"])
        plot_pdp(rf, X, feature_names, top_idx, config["significant_only"])

        anova_model = run_anova(df, output, config["input_categoricals"], config["input_numerics"])
        print(anova_model.summary2().tables[1][anova_model.pvalues < 0.05])

if __name__ == "__main__":
    main()
