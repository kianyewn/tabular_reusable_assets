import pandas as pd
from scipy import stats


def two_way_chi_square_test(data, categorical_feat_x, categorical_feat_y):
    observed = pd.crosstab(data[categorical_feat_x], data[categorical_feat_y])
    chi_s = stats.chi2_contingency(observed)
    return chi_s
