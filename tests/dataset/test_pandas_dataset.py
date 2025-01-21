from datetime import datetime
from unittest.mock import patch

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tabular_reusable_assets.datasets import DatasetKIND
from tabular_reusable_assets.datasets.data_catalog import Dataset


class Data(BaseModel):
    data: DatasetKIND = Field(..., discriminator="KIND")


class MutableDataset(BaseModel):
    model_config = ConfigDict(extra="allow")
    training_input: DatasetKIND = Field(..., discriminator="KIND")

    def add(self, name, dataset):
        setattr(self, name, Data.model_validate({"data": dataset}).data)


def test_read_pandas():
    dummy_dataset = MutableDataset(
        training_input={"KIND": "PandasCSVDataset", "path": "data/dummy.csv", "read_args": {"header": 0}}
    )
    dummy_dataset.add(
        "training_input2", {"KIND": "PandasCSVDataset", "path": "data/dummy.csv", "read_args": {"header": 0}}
    )

    dummy_pd = dummy_dataset.training_input.read()
    assert isinstance(dummy_pd, pd.DataFrame)

    dummy_pd = dummy_dataset.training_input2.read()
    print(dummy_dataset)
    assert isinstance(dummy_pd, pd.DataFrame)


@patch("tabular_reusable_assets.datasets.data_catalog.Dataset.write", return_value=None)
def test_get_latest_path(write):
    dummy_dataset = Dataset(
        dataset={"KIND": "PandasCSVDataset", "path": "data/9999-12-31/dummy.csv", "read_args": {"header": 0}}
    )
    assert dummy_dataset.try_get_latest_file() == "data/2025-01-01/dummy.csv"

    # test ability for dataset to recognize we should read latest file if date is 9999-12-31
    dummy_dataset.read()
    assert dummy_dataset.mock_dataset.path == "data/2025-01-01/dummy.csv"
    # check that path is still unchanged
    assert dummy_dataset.dataset.path == "data/9999-12-31/dummy.csv"

    # Writing should produce today'date
    today = datetime.now().strftime("%Y-%m-%d")
    dummy_dataset.write(dummy_dataset.read())
    todays_date = dummy_dataset.replace_date_in_path_with_today()
    assert todays_date == f"data/{today}/dummy.csv"
