from tabular_reusable_assets.datasets.pandas_dataset import PandasCSVDataset, PandasParquetDataset


DatasetKIND= PandasParquetDataset | PandasCSVDataset


__all__ = ["DatasetKIND"]
