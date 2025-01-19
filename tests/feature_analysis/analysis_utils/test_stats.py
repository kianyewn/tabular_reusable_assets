import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve

from tabular_reusable_assets.feature_analysis.analysis_utils import MonotonicityAnalysis, two_way_chi_square_test


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


def test_roc_curve(titanic_dataset):
    data = titanic_dataset["data"]
    # threshold is sorted in reverse
    fpr, tpr, thr = roc_curve(data["Survived"], y_score=data["Age"])
    assert np.allclose(sorted(thr, reverse=True), thr)
    assert max(fpr) == 1
    assert max(tpr) == 1

def test_ks(titanic_dataset):
    data = titanic_dataset["data"]
    stat = data.groupby("Age")["Survived"].value_counts().unstack().fillna(0).rename(columns={0: "neg", 1: "pos"})
    stat["neg_cnt"] = data[data["Survived"] == 0].shape[0]
    stat["pos_cnt"] = data[data["Survived"] == 1].shape[0]
    stat["neg_rate"] = stat["neg"] / stat["neg_cnt"]
    stat["pos_rate"] = stat["pos"] / stat["pos_cnt"]
    stat = stat.sort_index().reset_index()
    stat = stat.reset_index()
    manual_ks = max(abs(stat["neg_rate"].cumsum() - stat["pos_rate"].cumsum()))
    scipy_ks = ks_2samp(data.loc[data.Survived == 0, "Age"], data.loc[data.Survived == 1, "Age"])
    assert np.allclose(manual_ks, scipy_ks.statistic)
    ks = MonotonicityAnalysis.get_ks(y_col=data["Age"], y_true=data["Survived"])
    assert np.allclose(ks, scipy_ks.statistic)


def test_auc(titanic_dataset):
    data = titanic_dataset["data"]
    roc_true = roc_auc_score(data["Survived"], data["Age"])
    roc_true = max(1 - roc_true, roc_true)
    roc = MonotonicityAnalysis.get_auc(data["Age"], data["Survived"])
    assert np.allclose(roc, roc_true)

def test_gini(titanic_dataset):
    data = titanic_dataset["data"]
    roc_true = roc_auc_score(data["Survived"], data["Age"])
    roc_true = max(1 - roc_true, roc_true)
    gini_true = 2 * roc_true - 1
    gini = MonotonicityAnalysis.get_gini(data["Age"], data["Survived"])
    assert np.allclose(gini, gini_true)



