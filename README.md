# Statistics Tools

Sets of `scipy`, `statsmodels` and self-made tools for statistical analysis. Implemented for simplified access to them.

## Intervals

File: *intervals.py*. Includes realizations of confidence and statistical intervals:

- `zconfint` - confidence interval based on normal distribution;
- `zconfint_diff` - confidence interval based on normal distribution for the difference in means of two samples;
- `tconfint` - confidence interval based on Student t distribution;
- `tconfint_diff` - confidence interval based on Student t distribution for the difference in means of two samples;
- `var_confint` - confidence interval for sample variance;
- `bootstrap_statint` - statistical interval for a `stat` of a `sample` calculation using bootstrap sampling mechanism;
- `bootstrap_statint_diff` - statistical interval for a difference in `stat` of two samples calculation using bootstrap sampling mechanism;
- `proportion_confint` - Wilson's сonfidence interval for a proportion;
- `proportions_confint_diff_ind` - confidence interval for the difference of two independent proportions;
- `proportions_confint_diff_rel` - confidence interval for the difference of two related proportions.

## Hypotheses testing

File: *hypotheses.py*. Includes realizations of statistical tests:

- `permutation_test` - permutation test for a sample;
- `permutation_test_ind` - permutation test for two independent samples;
- `proportion_ztest` - Z-test for a proportion;
- `proportions_ztest_ind` - Z-test for two independent proportions;
- `proportions_ztest_rel` - Z-test for two related proportions.

Some of `scipy` and `statsmodels` methods can also be accessed.

**List of Scipy methods**: `chisquare`, `shapiro`, `ttest_1samp`, `ttest_ind`, `ttest_rel`, `median_test`, `wilcoxon`, `mannwhitneyu`.

**List of Statsmodels methods**: `sign_test`, `multipletests`.

## Correlations

File: *correlations.py*. Includes realizations of different types of correlation calculations:

- `matthews_correlation` - Matthews correlation;
- `cramers_v` - Cramer's V coefficient.

Some of `scipy` methods can also be accessed: `pearsonr`, `spearmanr`, `chi2_contingency`, `pointbiserialr`.
