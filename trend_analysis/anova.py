from statsmodels.formula.api import ols
from itertools import combinations

def run_anova(df, output, input_categoricals, input_numerics):
    results = {}

    cat_terms = [f'C(Q("{col}"))' for col in input_categoricals]
    num_terms = [f'Q("{col}")' for col in input_numerics]
    all_terms = cat_terms + num_terms

    max_depth = len(all_terms)

    for depth in range(1, max_depth + 1):
        formula_terms = []
        for combo in combinations(all_terms, depth):
            formula_terms.append(':'.join(combo))

        formula = f'Q("{output}") ~ ' + ' + '.join(formula_terms)
        model = ols(formula, data=df).fit()
        results[depth] = model

    return results

