import pandas as pd
from pydantic import BaseModel, Field

from tabular_reusable_assets.datasets import DatasetKIND


class DataFactory(BaseModel):
    training_input: DatasetKIND = Field(..., discriminator="KIND")



def test_read_pandas():
    dummy_dataset = DataFactory(training_input={"KIND": "PandasCSVDataset", "path": "data/dummy.csv"})
    dummy_pd = dummy_dataset.training_input.read()
    print(dummy_pd.shape)
    assert isinstance(dummy_pd, pd.DataFrame)
