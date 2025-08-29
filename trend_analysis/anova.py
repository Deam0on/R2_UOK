from statsmodels.formula.api import ols
from itertools import combinations

def run_anova(df, output, input_categoricals, input_numerics):
    import pandas as pd
    import os
    os.makedirs('output', exist_ok=True)
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
        # Export ANOVA table as CSV for each depth
        anova_table = model.summary2().tables[1]
        anova_table.to_csv(f'output/anova_{output}_depth{depth}.csv')

    return results

