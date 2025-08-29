"""
ANOVA analysis utilities for statistical modeling and result output.
"""

import logging
import os
from itertools import combinations

import pandas as pd
from statsmodels.formula.api import ols


def run_anova(df, output, input_categoricals, input_numerics, output_dir="output"):
    """
    Perform ANOVA analysis on the given DataFrame and export the results.

    Parameters:
    - df: The input DataFrame containing the data.
    - output: The name of the output variable for the ANOVA analysis.
    - input_categoricals: A list of categorical input variable names.
    - input_numerics: A list of numeric input variable names.
    - output_dir: Directory where to save output files. Default is "output".

    Returns:
    - results: A dictionary containing the fitted model for each depth of interaction.
    
    Raises:
    - ValueError: If invalid inputs are provided.
    - OSError: If output directory cannot be created.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        if output not in df.columns:
            raise ValueError(f"Output variable '{output}' not found in DataFrame columns")
        
        if not input_categoricals and not input_numerics:
            raise ValueError("No input variables provided")
        
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created/verified output directory: {output_dir}")
        except OSError as e:
            logger.exception(f"Failed to create output directory: {output_dir}")
            raise OSError(f"Cannot create output directory: {output_dir}") from e
        
        results = {}

        cat_terms = [f'C(Q("{col}"))' for col in input_categoricals if col in df.columns]
        num_terms = [f'Q("{col}")' for col in input_numerics if col in df.columns]
        all_terms = cat_terms + num_terms
        
        if not all_terms:
            logger.warning("No valid terms found for ANOVA analysis")
            return results

        max_depth = len(all_terms)
        logger.info(f"Running ANOVA analysis for {output} with {max_depth} maximum interaction depth")

        for depth in range(1, max_depth + 1):
            try:
                logger.debug(f"Processing ANOVA depth {depth}")
                formula_terms = []
                for combo in combinations(all_terms, depth):
                    formula_terms.append(":".join(combo))

                formula = f'Q("{output}") ~ ' + " + ".join(formula_terms)
                logger.debug(f"ANOVA formula for depth {depth}: {formula}")
                
                model = ols(formula, data=df).fit()
                results[depth] = model
                
                # Export ANOVA table as CSV for each depth
                try:
                    anova_table = model.summary2().tables[1]
                    output_file = os.path.join(output_dir, f"anova_{output}_depth{depth}.csv")
                    anova_table.to_csv(output_file)
                    logger.debug(f"Saved ANOVA table for depth {depth} to: {output_file}")
                except Exception as e:
                    logger.exception(f"Failed to save ANOVA table for depth {depth}: {e}")
                    
            except Exception as e:
                logger.exception(f"Failed to fit ANOVA model for depth {depth}: {e}")
                continue

        logger.info(f"ANOVA analysis completed for {output}. Generated {len(results)} models.")
        return results
        
    except Exception as e:
        logger.exception(f"Error in run_anova: {e}")
        print(f"Error in run_anova: {e}")
        return None
