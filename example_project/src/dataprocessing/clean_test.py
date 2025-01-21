from tabular_reusable_assets.datasets.data_catalog import Dataset


class TestCleaner:
    def __init__(self, dc_config: dict):
        self.dc_config = dc_config

    def clean_data(self):
        clean_test = Dataset(**self.dc_config["test_raw"]).read()
        return clean_test
