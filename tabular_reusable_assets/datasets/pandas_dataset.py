import typing as T

import pandas as pd

from tabular_reusable_assets.datasets.core import Reader, Writer


class PandasParquetDataset(Reader, Writer):
    KIND: T.Literal["PandasParquetDataset"] = "PandasParquetDataset"
    path: str
    read_args: T.Dict[str, T.Any] = None
    write_args: T.Dict[str, T.Any] = None

    _DEFAULT_READ_ARGS: T.Dict[str, T.Any] = {}
    _DEFAULT_WRITE_ARGS: T.Dict[str, T.Any] = {}

    def model_post_init(self, _context: T.Any = None) -> None:
        self._read_args = {**self._DEFAULT_READ_ARGS, **(self.read_args or {})}
        self._write_args = {**self._DEFAULT_WRITE_ARGS, **(self.write_args or {})}

    def read(self, **read_args) -> pd.DataFrame:
        self._read_args.update(read_args)
        return pd.read_parquet(self.path, **self._read_args)

    def write(self, data: pd.DataFrame, index: bool = False, **write_args):
        self._write_args.update(write_args)
        return data.to_parquet(self.path, index=index, **self._write_args)


class PandasCSVDataset(Reader, Writer):
    KIND: T.Literal["PandasCSVDataset"] = "PandasCSVDataset"
    path: str
    read_args: T.Dict[str, T.Any] = None
    write_args: T.Dict[str, T.Any] = None

    def model_post_init(self, _context: T.Any = None) -> None:
        self._read_args = {**self._DEFAULT_READ_ARGS, **(self.read_args or {})}
        self._write_args = {**self._DEFAULT_WRITE_ARGS, **(self.write_args or {})}

    def read(self, **read_args) -> pd.DataFrame:
        self._read_args.update(read_args)
        return pd.read_csv(self.path, **self._read_args)

    def write(self, data: pd.DataFrame, index: bool = False, **write_args):
        self._write_args.update(write_args)
        return data.to_csv(self.path, index=index, **self._write_args)


if __name__ == "__main__":
    dummy_dataset = PandasCSVDataset(path="data/dummy.csv")
    dummy_pd = dummy_dataset.read()
    assert isinstance(dummy_pd, pd.DataFrame)
    parquet_dataset = PandasParquetDataset(path="data/dummy.parquet")
    parquet_dataset.write(dummy_pd)
