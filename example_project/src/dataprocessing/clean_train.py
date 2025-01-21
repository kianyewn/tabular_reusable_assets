from tabular_reusable_assets.datasets.data_catalog import Dataset


class TrainCleaner:
    def __init__(self, dc_config: dict):
        self.dc_config = dc_config

    def clean_data(self):
        train = Dataset(**self.dc_config["train_raw"]).read()
        return train
