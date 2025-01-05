import numpy as np
import pandas as pd


def test_multi_index_from_value_counts():
    sample = pd.DataFrame({"col1": [0, 0, 1, 1], "col2": [3, 4, 5, 6]})

    multi_index = sample.groupby(["col1"])["col2"].value_counts().index

    assert np.all(multi_index == pd.MultiIndex.from_tuples([(0, 3), (0, 4), (1, 5), (1, 6)]))


def test_unstack():
    sample = pd.DataFrame({"col1": [0, 0, 1, 1], "col2": [3, 4, 5, 6]})
    vc = sample.groupby(["col1"]).value_counts().unstack().fillna(0)

    # unstack will create columns from all the unique values of value_counts()
    assert list(vc.columns) == [3, 4, 5, 6]
