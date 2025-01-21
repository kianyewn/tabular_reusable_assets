import typing as T
from datetime import datetime

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_reusable_assets.utils.file_helper import FileHelper

from .pandas_dataset import PandasCSVDataset, PandasParquetDataset


DatasetKIND = PandasCSVDataset | PandasParquetDataset


def is_string_template(path: str):
    return isinstance(path, str) and "{" in path


class Dataset(BaseModel):
    dataset: DatasetKIND = Field(..., discriminator="KIND")
    mock_dataset: DatasetKIND = None

    def read(self, **read_args) -> pd.DataFrame:
        if "9999-12-31" in self.dataset.path:
            dataset = self.dataset.model_copy(update={"path": FileHelper.get_latest_file(self.dataset.path)})
            logger.info(f"Using latest file: `{dataset.path}` since `9999-12-31` is provided.")
            self.mock_dataset = dataset
            return dataset.read(**read_args)
        return self.dataset.read(**read_args)

    def replace_date_in_path_with_today(self):
        todays_date = datetime.now().strftime("%Y-%m-%d")
        new_path = FileHelper.replace_date_in_path(self.dataset.path, replacement_date=todays_date)
        return new_path

    def write(self, data: pd.DataFrame, **write_args):
        if "9999-12-31" in self.dataset.path:
            new_path = self.replace_date_in_path_with_today()
            logger.info(f"Replacing path: `{self.dataset.path}` with `{new_path}`")
            dataset = self.dataset.model_copy(update={"path": new_path})
            self.mock_dataset = dataset

            logger.info(f"Updated path to today's date: `{dataset.path}`")
            FileHelper.try_mkdir(dataset_path=dataset.path)
            return dataset.write(data, **write_args)

        FileHelper.try_mkdir(dataset_path=self.dataset.path)
        return self.dataset.write(data, **write_args)

    def try_get_latest_file(self):
        return self.dataset.try_get_latest_file()


class DataCatalog(BaseModel):
    datasets: T.Union[T.Any, T.Dict[str, T.Dict[str, T.Any]]] = None
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    def validate_and_convert_datasets(cls, values):
        datasets = values.get("datasets", {}).copy()
        for key, dataset in datasets.items():
            if isinstance(dataset, dict):  # Convert dict to the appropriate DatasetKIND
                kind = dataset.get("KIND")
                if kind == "PandasCSVDataset":
                    datasets[key] = PandasCSVDataset(**dataset)
                elif kind == "PandasParquetDataset":
                    datasets[key] = PandasParquetDataset(**dataset)
                else:
                    raise ValueError(f"Unsupported dataset kind: {kind}")
        values["datasets"] = datasets
        return values

    def model_post_init(self, _context):
        for key, dataset in self.datasets.items():
            setattr(self, key, dataset)
        return

    def __getitem__(self, key: str):
        return self.datasets[key]

    def load(self, name: str):
        return self.datasets[name].read()


class DataCatalog2:
    def __init__(self, datasets):
        self.datasets = datasets
        self.init_datasets()

    def init_datasets(self):
        for key, dataset in self.datasets.items():
            if isinstance(dataset, dict):  # Convert dict to the appropriate DatasetKIND
                kind = dataset.get("KIND")
                if kind == "PandasCSVDataset":
                    setattr(self, key, PandasCSVDataset(**dataset))
                elif kind == "PandasParquetDataset":
                    setattr(self, key, PandasParquetDataset(**dataset))
                else:
                    raise ValueError(f"Unsupported dataset kind: {kind}")
        return

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key: {key} does not exist in config")
