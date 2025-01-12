from typing import List

import numpy as np
import pandas as pd


def get_equal_width_bins(data_feat: pd.Series, nbins=10) -> List:
    min_value = min(data_feat) * 0.95
    max_value = max(data_feat)
    delta = max_value - min_value
    breaks = [min_value + 0.1 * i * delta for i in range(nbins + 1)]
    return breaks


def get_quantile_bins(data_feat: pd.Series, nbins=10) -> List:
    quantiles = np.linspace(start=0, stop=1, num=nbins + 1)
    breaks = [data_feat.quantile(q) for q in quantiles]
    breaks[0] = breaks[0] * 0.99 if breaks[0] > 0 else breaks[0] * 1.01
    # print(data_feat)
    # print(breaks, quantiles)
    return breaks


def bin_feat(data, feat, breaks, include_lowest=False, precision=3, duplicates="drop", return_categorical=False):
    if not return_categorical:
        # return str
        return pd.cut(
            data[feat], bins=breaks, include_lowest=include_lowest, precision=precision, duplicates=duplicates
        ).astype(str)
    return pd.cut(data[feat], bins=breaks, include_lowest=include_lowest, precision=precision, duplicates=duplicates)


def qcut(
    data,
    feat,
    nbins,
    include_lowest=False,
    precision=3,
    duplicates="drop",
    return_categorical=False,
    return_bins=False,
):
    breaks = get_quantile_bins(data[feat], nbins=nbins)

    feat_binned = bin_feat(
        data=data,
        feat=feat,
        breaks=breaks,
        include_lowest=include_lowest,
        precision=precision,
        duplicates=duplicates,
        return_categorical=return_categorical,
    )
    if not return_bins:
        return feat_binned
    else:
        return feat_binned, breaks


def equal_width_cut(
    data,
    feat,
    nbins,
    include_lowest=False,
    precision=3,
    duplicates="drop",
    return_categorical=False,
    return_bins=False,
):
    breaks = get_equal_width_bins(data[feat], nbins=nbins)
    feat_binned = bin_feat(
        data=data,
        feat=feat,
        breaks=breaks,
        include_lowest=include_lowest,
        precision=precision,
        duplicates=duplicates,
        return_categorical=return_categorical,
    )

    if not return_bins:
        return feat_binned
    else:
        return feat_binned, breaks


def calculate_feat_psi_table(
    left_df, right_df, feat, left_df_name="A", right_df_name="B", eps=np.finfo(float).eps, rounding=True
):
    """Return the table used to calculate the psi to inspect individual bin psi"""
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
    if rounding:
        vc["psi"] = round(vc["psi"], 5)
    return vc[columns]


def calculate_feat_psi(left_df, right_df, feat, left_df_name="A", right_df_name="B", eps=np.finfo(float).eps):
    """Returns only two columns for report"""
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

    return vc[["var", "sum_psi"]].drop_duplicates()


class Binning:
    def __init__(
        self,
        numerical_features,
        nbins=10,
        new_col=False,
    ):
        self.numerical_features = numerical_features
        self.feat_bins = {}
        self.nbins = nbins
        self.new_col = new_col

    def fit(self, X, y=None):
        for feat in self.numerical_features:
            breaks = get_quantile_bins(X[feat], nbins=self.nbins)
            # prevent outliers
            breaks[0] = -np.inf
            breaks[-1] = np.inf
            self.feat_bins[feat] = breaks
        return

    def transform(self, X, y=None):
        X = X.copy()
        for feat in self.numerical_features:
            col_name = feat if not self.new_col else f"{feat}_bin"
            X[col_name] = bin_feat(data=X, feat=feat, breaks=self.feat_bins[feat])
        return X
