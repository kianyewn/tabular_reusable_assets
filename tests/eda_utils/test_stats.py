import numpy as np
import pandas as pd
from scipy import stats

from tabular_reusable_assets.feature_analysis.eda_utils import two_way_chi_square_test


def test_two_way_chi_square():
    np.random.seed(10)

    # Sample data randomly at fixed probabilities
    voter_race = np.random.choice(
        a=["asian", "black", "hispanic", "other", "white"], p=[0.05, 0.15, 0.25, 0.05, 0.5], size=1000
    )

    voter_party = np.random.choice(a=["democrat", "independent", "republican"], p=[0.4, 0.2, 0.4], size=1000)
    data = pd.DataFrame({"voter_race": voter_race, "voter_party": voter_party})

    chi_s = two_way_chi_square_test(data=data, categorical_feat_x="voter_race", categorical_feat_y="voter_party")
    assert isinstance(chi_s, stats.contingency.Chi2ContingencyResult)
    res_dict = chi_s._asdict()
    res = {
        "statistic": 7.169321280162059,
        "pvalue": 0.518479392948842,
        "dof": 8,
        "expected_freq": np.array(
            [
                [23.82, 11.16, 25.02],
                [61.138, 28.644, 64.218],
                [99.647, 46.686, 104.667],
                [15.086, 7.068, 15.846],
                [197.309, 92.442, 207.249],
            ]
        ),
    }
    assert all(np.allclose(res[key], res_dict[key]) for key in res.keys())

    # to take note: need to take into consideration the order of the columns
    wrong_expected = np.outer(pd.Series(voter_race).value_counts(), pd.Series(voter_party).value_counts()) / 1000
    wrong_expected_res = np.array(
        [
            [207.249, 197.309, 92.442],
            [104.667, 99.647, 46.686],
            [64.218, 61.138, 28.644],
            [25.02, 23.82, 11.16],
            [15.846, 15.086, 7.068],
        ]
    )
    assert np.allclose(wrong_expected, wrong_expected_res)

    #### Manual Calculation for concept checking ####
    # manual chi square calculation
    n_samples = data.shape[0]
    contingency_table = pd.crosstab(data["voter_race"], data["voter_party"], margins=True)
    # sorted according to alphabetical ordering
    assert contingency_table.columns.tolist() == ["democrat", "independent", "republican", "All"]
    assert contingency_table.index.tolist() == ["asian", "black", "hispanic", "other", "white", "All"]
    assert contingency_table.shape == (
        data["voter_race"].nunique() + 1,
        data["voter_party"].nunique() + 1,
    )  # +1 for margin
    observed = contingency_table.iloc[:-1, :-1]  # exclude margin column
    assert observed.columns.tolist() == ["democrat", "independent", "republican"]
    assert observed.index.tolist() == ["asian", "black", "hispanic", "other", "white"]

    # calculate P(A,B) = P(A) * P(B)
    expected = (
        np.outer(contingency_table.loc[:, "All"].iloc[:-1], contingency_table.loc["All", :].iloc[:-1]) / n_samples
    )
    assert np.allclose(expected, chi_s.expected_freq)

    # calculate chi_s values manually
    chi_s_value = ((observed.values - expected) ** 2 / (expected)).sum()
    assert chi_s_value == chi_s.statistic
