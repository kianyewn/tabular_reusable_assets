from example_project.src.dataprocessing.clean_train import TrainCleaner
from tabular_reusable_assets.datasets.data_catalog import Dataset
from tabular_reusable_assets.utils.config_helper import ConfigYAML


if __name__ == "__main__":
    dc_config = ConfigYAML.load("example_project/config/base/catalog/data_catalog.yaml")
    train_loader = TrainCleaner(dc_config)
    cleaned_train = train_loader.clean_data()
    # write dataset
    Dataset(**dc_config["train_processed"]).write(cleaned_train)
