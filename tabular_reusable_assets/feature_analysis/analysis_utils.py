import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def two_way_chi_square_test(data, categorical_feat_x, categorical_feat_y):
    observed = pd.crosstab(data[categorical_feat_x], data[categorical_feat_y])
    chi_s = stats.chi2_contingency(observed)
    return chi_s


def get_feat_psi(left_df, right_df, feat, left_df_name="A", right_df_name="B", eps=np.finfo(float).eps):
    A = left_df_name
    B = right_df_name
    left_vc = left_df[feat].value_counts(dropna=False, normalize=True).rename(A).reset_index()
    right_vc = right_df[feat].value_counts(dropna=False, normalize=True).rename(B).reset_index()
    vc = left_vc.merge(right_vc, on=[feat], how="outer").fillna(0)
    vc["var"] = feat
    vc = vc.sort_values(by=feat).rename(columns={feat: "bin"})

    vc[f"{A}-{B}"] = vc[A] - vc[B]
    vc[f"ln({A}/{B})"] = np.log(vc[A] / (vc[B] + eps))
    vc["psi"] = vc[f"{A}-{B}"] * vc[f"ln({A}/{B})"]
    vc["sum_psi"] = vc["psi"].sum()

    columns = ["var", "bin"] + [col for col in vc.columns if col not in ["var", "bin"]]
    return vc[columns]


class MonotonicityAnalysis:
    def get_ks(y_col, y_true):
        # threshold is sorted in reverse
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_col)
        ks = max(abs(tpr - fpr))
        return ks

    def get_gini(y_col, y_true):
        # Gini scaled from 0 to 1
        auc = roc_auc_score(y_true=y_true, y_score=y_col)
        auc = max(auc, 1 - auc)
        return 2 * auc - 1

    def get_auc(y_col, y_true):
        auc = roc_auc_score(y_true=y_true, y_score=y_col)
        return max(auc, 1 - auc)