import numpy as np
from scipy import stats


def zconfint(sample, sigma=None, alpha=0.05):
    '''Confidence interval based on normal distribution for sample mean.'''
    mean = np.mean(sample)
    n = len(sample)

    if not sigma:
        sigma = np.std(sample, ddof=1)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = mean - z * sigma / np.sqrt(n)
    right_boundary = mean + z * sigma / np.sqrt(n)

    return left_boundary, right_boundary


def zconfint_diff(sample1, sample2, sigma1=None, sigma2=None, alpha=0.05):
    '''Confidence interval based on normal distribution for
    the difference in means of two samples.'''
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    if not sigma1:
        sigma1 = np.std(sample1, ddof=1)

    if not sigma2:
        sigma2 = np.std(sample2, ddof=1)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = (mean1 - mean2) - z * np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + z * np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)

    return left_boundary, right_boundary


def tconfint(sample, alpha=0.05):
    '''Confidence interval based on Student t distribution for sample mean.'''
    mean = np.mean(sample)
    S = np.std(sample, ddof=1)
    n = len(sample)

    t = stats.t.ppf(1 - alpha / 2, n - 1)
    left_boundary = mean - t * S / np.sqrt(n)
    right_boundary = mean + t * S / np.sqrt(n)

    return left_boundary, right_boundary


def tconfint_diff(sample1, sample2, alpha=0.05):
    '''Confidence interval based on Student t distribution for
    the difference in means of two samples.'''
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    s1 = np.std(sample1, ddof=1)
    s2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)

    sem1 = np.var(sample1) / (n1 - 1)
    sem2 = np.var(sample2) / (n2 - 1)
    semsum = sem1 + sem2
    z1 = (sem1 / semsum) ** 2 / (n1 - 1)
    z2 = (sem2 / semsum) ** 2 / (n2 - 1)
    dof = 1 / (z1 + z2)

    t = stats.t.ppf(1 - alpha / 2, dof)
    left_boundary = (mean1 - mean2) - t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)

    return left_boundary, right_boundary


def var_confint(sample, alpha=0.05):
    '''Confidence interval for sample variance.'''
    var = np.var(sample, ddof=1)
    n = len(sample)

    chi2_l = stats.chi2.ppf(1 - alpha / 2, n - 1)
    chi2_r = stats.chi2.ppf(alpha / 2, n - 1)
    left_boundary = ((n - 1) * var) / chi2_l
    right_boundary = ((n - 1) * var) / chi2_r

    return left_boundary, right_boundary


def bootstrap_statint(sample, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Statistical interval for a `stat` of a `sample` calculation
    using bootstrap sampling mechanism. `stat` is a numpy function
    like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices = np.random.randint(0, len(sample), (n_samples, len(sample)))
    samples = sample[indices]

    stat_scores = stat(samples, axis=1)
    boundaries = np.percentile(stat_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def bootstrap_statint_diff(sample1, sample2, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Statistical interval for a difference in `stat` of two samples
    calculation using bootstrap sampling mechanism. `stat` is a numpy
    function like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices1 = np.random.randint(0, len(sample1), (n_samples, len(sample1)))
    indices2 = np.random.randint(0, len(sample2), (n_samples, len(sample2)))
    samples1 = sample1[indices1]
    samples2 = sample2[indices2]

    stat_scores1 = stat(samples1, axis=1)
    stat_scores2 = stat(samples2, axis=1)
    stat_scores_diff = stat_scores1 - stat_scores2
    boundaries = np.percentile(stat_scores_diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def proportion_confint(sample, alpha=0.05):
    '''Wilson\'s сonfidence interval for a proportion.'''
    p = np.mean(sample)
    n = len(sample)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                            - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))
    right_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                             + z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))

    return left_boundary, right_boundary


def proportions_confint_diff_ind(sample1, sample2, alpha=0.05):
    '''Confidence interval for the difference of two independent proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    return left_boundary, right_boundary


def proportions_confint_diff_rel(sample1, sample2, alpha=0.05):
    '''Confidence interval for the difference of two related proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([obs[0] == 1 and obs[1] == 0 for obs in sample])
    g = sum([obs[0] == 0 and obs[1] == 1 for obs in sample])

    left_boundary = (f - g) / n - z * np.sqrt((f + g) / n ** 2 - ((f - g) ** 2) / n ** 3)
    right_boundary = (f - g) / n + z * np.sqrt((f + g) / n ** 2 - ((f - g) ** 2) / n ** 3)

    return left_boundary, right_boundary
