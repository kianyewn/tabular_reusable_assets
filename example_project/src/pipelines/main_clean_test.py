from example_project.src.dataprocessing.clean_test import TestCleaner
from tabular_reusable_assets.datasets.data_catalog import Dataset
from tabular_reusable_assets.utils.config_helper import ConfigYAML


if __name__ == "__main__":
    dc_config = ConfigYAML.load("example_project/config/base/catalog/data_catalog.yaml")
    test_loader = TestCleaner(dc_config)
    cleaned_test = test_loader.clean_data()
    # write dataset
    Dataset(**dc_config["test_processed"]).write(cleaned_test)
