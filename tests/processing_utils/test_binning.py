import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from tabular_reusable_assets import processing_utils
from tabular_reusable_assets.feature_analysis.processing_utils import Binning


def test_equal_width_bin(titanic_dataset):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]

    breaks = processing_utils.get_equal_width_bins(data[feature_columns[0]], nbins=10)
    # additional for start
    assert len(breaks) == 10 + 1
    assert np.allclose(np.diff(breaks), np.diff(breaks)[0])


def test_quantile_bin(titanic_dataset):
    data = titanic_dataset["data"]

    quantile_bins = processing_utils.get_quantile_bins(data["Age"], nbins=4)
    data["BAge"] = processing_utils.bin_feat(data, "Age", quantile_bins, duplicates="drop")
    vc1 = data["BAge"].value_counts(dropna=False, normalize=True)

    q_bins = pd.qcut(data["Age"], q=4, duplicates="drop").astype(str)
    vc2 = q_bins.value_counts(dropna=False, normalize=True)

    # additional for start
    assert len(quantile_bins) == 4 + 1
    assert np.allclose(vc1.values, vc2.values)


def test_psi_calculation_table(titanic_dataset):
    data = titanic_dataset["data"]
    # feature_columns = titanic_dataset["feature_columns"]
    # target_column = titanic_dataset["target_column"]

    data["qAge"] = processing_utils.qcut(data, "Age", nbins=10)
    data2 = data.sample(frac=0.5, random_state=99)

    indiv_psi_table = processing_utils.calculate_feat_psi_table(
        left_df=data, right_df=data2, feat="qAge", rounding=False
    )
    assert list(indiv_psi_table.columns) == ["var", "bin", "A", "B", "A-B", "ln(A/B)", "psi", "sum_psi"]

    # check element wise operation
    row = indiv_psi_table.iloc[0]
    assert np.allclose(row["A-B"] * row["ln(A/B)"], row["psi"])


def test_psi_calculation(titanic_dataset):
    data = titanic_dataset["data"]
    # feature_columns = titanic_dataset["feature_columns"]
    # target_column = titanic_dataset["target_column"]

    data["qAge"] = processing_utils.qcut(data, "Age", nbins=10)
    data2 = data.sample(frac=0.5, random_state=99)

    indiv_psi_table = processing_utils.calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert list(indiv_psi_table.columns) == ["var", "sum_psi"]


def test_psi_detect_different_distribution(titanic_dataset):
    data = titanic_dataset["data"]

    data["qAge"] = processing_utils.qcut(data, "Age", nbins=10)

    # very small sample, psi should be high
    data2 = data.sample(1, random_state=99)
    indiv_psi_table = processing_utils.calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert indiv_psi_table["sum_psi"].unique()[0] > 0.2

    # same sample, psi should be small close to 0
    data2 = data.sample(frac=1, random_state=99)
    indiv_psi_table = processing_utils.calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert indiv_psi_table["sum_psi"].unique()[0] == 0


def test_psi_good_and_bad_dummy():
    df1 = pd.DataFrame({"col1": ["a"] * 2 + ["b"] * 10})
    df2 = pd.DataFrame({"col1": ["a"] * 2 + ["b"] * 10})
    df3 = pd.DataFrame({"col1": ["a"] * 7 + ["b"] * 5})
    assert processing_utils.calculate_feat_psi(df1, df2, feat="col1")["sum_psi"].iloc[0] == 0
    assert np.allclose(
        processing_utils.calculate_feat_psi(df1, df3, feat="col1")["sum_psi"].iloc[0], 0.810795895439714
    )


def test_binner_with_psi(titanic_dataset):
    data = titanic_dataset["data"]
    train = data
    val = data
    numerical_features = ["Age"]
    binner = Binning(numerical_features=numerical_features, new_col=True)
    binner.fit(train)
    train2 = binner.transform(train)
    val2 = binner.transform(val)
    assert_series_equal(train2["Age_bin"], val2["Age_bin"])

    # calculate psi based on bin
    psi = processing_utils.calculate_feat_psi(train2, val2, feat="Age_bin")
    assert psi["sum_psi"].iloc[0] == 0
