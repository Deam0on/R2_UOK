"""
ANOVA analysis utilities for statistical modeling and result output.
"""

from itertools import combinations

from statsmodels.formula.api import ols


def run_anova(df, output, input_categoricals, input_numerics, output_dir="output"):
    """
    Perform ANOVA analysis on the given DataFrame and export the results.

    Parameters:
    - df: The input DataFrame containing the data.
    - output: The name of the output variable for the ANOVA analysis.
    - input_categoricals: A list of categorical input variable names.
    - input_numerics: A list of numeric input variable names.

    Returns:
    - results: A dictionary containing the fitted model for each depth of interaction.
    """
    import os
    import pandas as pd
    try:
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        cat_terms = [f'C(Q("{col}"))' for col in input_categoricals]
        num_terms = [f'Q("{col}")' for col in input_numerics]
        all_terms = cat_terms + num_terms

        max_depth = len(all_terms)

        for depth in range(1, max_depth + 1):
            formula_terms = []
            for combo in combinations(all_terms, depth):
                formula_terms.append(":".join(combo))

            formula = f'Q("{output}") ~ ' + " + ".join(formula_terms)
            model = ols(formula, data=df).fit()
            results[depth] = model
        # Export ANOVA table as CSV for each depth
        anova_table = model.summary2().tables[1]
        anova_table.to_csv(os.path.join(output_dir, f"anova_{output}_depth{depth}.csv"))

        return results
    except Exception as e:
        print(f"Error in run_anova: {e}")
        return None
