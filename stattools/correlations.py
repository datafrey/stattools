import numpy as np
from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr


def matthews_correlation(contingency_table):
    '''Matthews correlation.'''
    a, b = contingency_table[0]
    c, d = contingency_table[1]

    n = np.sum(contingency_table)
    acabn = (a + c) * (a + b) / n
    accdn = (a + c) * (c + d) / n
    bdabn = (b + d) * (a + b) / n
    bdcdn = (b + d) * (c + d) / n
    if n < 40 or np.any(np.array([acabn, accdn, bdabn, bdcdn]) < 5):
        raise ValueError('Contingency table isn\'t suitable for Matthews correlation calculation.')

    p_value = stats.chi2_contingency(contingency_table)[1]
    corr = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return corr, p_value


def cramers_v(contingency_table):
    '''Cramer\'s V coefficient.'''
    n = np.sum(contingency_table)
    ct_nrows, ct_ncols = contingency_table.shape
    if n < 40 or np.sum(contingency_table < 5) / (ct_nrows * ct_ncols) > 0.2:
        raise ValueError('Contingency table isn\'t suitable for Cramers\'s V coefficient calculation.')

    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    corr = np.sqrt(chi2 / (n * (min(ct_nrows, ct_ncols) - 1)))
    return corr, p_value
