from statsmodels.formula.api import ols

def run_anova(df, output, input_categoricals, input_numerics):
    terms = [f'C(Q("{col}"))' for col in input_categoricals] + [f'Q("{col}")' for col in input_numerics]
    base_formula = f'Q("{output}") ~ ' + " + ".join(terms)
    model = ols(base_formula, data=df).fit()
    return model
