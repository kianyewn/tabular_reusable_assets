import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from tabular_reusable_assets.feature_analysis import processing_utils
from tabular_reusable_assets.feature_analysis.data_drift_utils import calculate_feat_psi, calculate_feat_psi_table


def test_psi_calculation_table(titanic_dataset):
    data = titanic_dataset["data"]
    # feature_columns = titanic_dataset["feature_columns"]
    # target_column = titanic_dataset["target_column"]

    data["qAge"] = processing_utils.qcut(data, "Age", nbins=10)
    data2 = data.sample(frac=0.5, random_state=99)

    indiv_psi_table = calculate_feat_psi_table(left_df=data, right_df=data2, feat="qAge", rounding=False)
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

    indiv_psi_table = calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert list(indiv_psi_table.columns) == ["var", "sum_psi"]


def test_psi_detect_different_distribution(titanic_dataset):
    data = titanic_dataset["data"]

    data["qAge"] = processing_utils.qcut(data, "Age", nbins=10)

    # very small sample, psi should be high
    data2 = data.sample(1, random_state=99)
    indiv_psi_table = calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert indiv_psi_table["sum_psi"].unique()[0] > 0.2

    # same sample, psi should be small close to 0
    data2 = data.sample(frac=1, random_state=99)
    indiv_psi_table = calculate_feat_psi(left_df=data, right_df=data2, feat="qAge")
    assert indiv_psi_table["sum_psi"].unique()[0] == 0


def test_psi_good_and_bad_dummy():
    df1 = pd.DataFrame({"col1": ["a"] * 2 + ["b"] * 10})
    df2 = pd.DataFrame({"col1": ["a"] * 2 + ["b"] * 10})
    df3 = pd.DataFrame({"col1": ["a"] * 7 + ["b"] * 5})
    assert calculate_feat_psi(df1, df2, feat="col1")["sum_psi"].iloc[0] == 0
    assert np.allclose(calculate_feat_psi(df1, df3, feat="col1")["sum_psi"].iloc[0], 0.810795895439714)


def test_binner_with_psi(titanic_dataset):
    data = titanic_dataset["data"]
    train = data
    val = data
    numerical_features = ["Age"]
    binner = processing_utils.Binning(numerical_features=numerical_features, new_col=True)
    binner.fit(train)
    train2 = binner.transform(train)
    val2 = binner.transform(val)
    assert_series_equal(train2["Age_bin"], val2["Age_bin"])

    # calculate psi based on bin
    psi = calculate_feat_psi(train2, val2, feat="Age_bin")
    assert psi["sum_psi"].iloc[0] == 0
