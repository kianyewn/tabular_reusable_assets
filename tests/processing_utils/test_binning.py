import numpy as np
import pandas as pd

from tabular_reusable_assets import processing_utils


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
