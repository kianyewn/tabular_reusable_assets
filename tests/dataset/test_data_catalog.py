from datetime import datetime

import pandas as pd

from tabular_reusable_assets.datasets.data_catalog import DataCatalog
from tabular_reusable_assets.datasets.datasets_utils import FilePath, PathBuilder, PathTemplates
from tabular_reusable_assets.datasets.pandas_dataset import PandasCSVDataset, PandasParquetDataset
from tabular_reusable_assets.utils.config_helper import ConfigYAML


def test_pandas_csv_dataset():
    data = PandasCSVDataset(path="data/dummy.csv")
    pd_data = data.read()
    assert isinstance(pd_data, pd.DataFrame)


def test_pandas_parquet_dataset():
    data = PandasParquetDataset(path="data/dummy.parquet")
    pd_data = data.read()
    assert isinstance(pd_data, pd.DataFrame)


def test_datacatalog():
    dummy_dataset = DataCatalog(
        datasets={
            "training_input": {"KIND": "PandasCSVDataset", "path": "data/dummy.csv", "read_args": {"header": 0}},
            "training_input2": {"KIND": "PandasCSVDataset", "path": "data/dummy.csv", "read_args": {"header": 0}},
        }
    )
    dummy_pd = dummy_dataset["training_input"].read()
    assert isinstance(dummy_pd, pd.DataFrame)

    # test programmaticall
    dummy_pd = dummy_dataset.training_input.read()
    assert isinstance(dummy_pd, pd.DataFrame)

    # Load form name
    dummy_pd = dummy_dataset.load("training_input")
    assert isinstance(dummy_pd, pd.DataFrame)

    # Load form name
    dummy_pd = dummy_dataset.load("training_input2")
    assert isinstance(dummy_pd, pd.DataFrame)


def test_path_builder_from_python_templates():
    env = "prd"  # from user input
    date = "2011-12-12"
    global_config = {
        "s3_bucket": "s3_bucket",
        "project_name": "project_name",
        "business_objective": "business_objective",
        "version": "0.0.1-beta",
        "env": "dev",
        "datasets": {
            "training_input_file_name": "model_input_data.parquet",
            "inference_input_file_name": "inference_input_data.parquet",
            "saved_model_file_name": "saved_model.pickle",
            "inference_output_file_name": "inference_output.parquet",
        },
    }

    path_builder = PathBuilder(config=global_config)
    version = "0.0.0"
    # from processing
    processing_date = datetime.now().strftime("%Y-%m-%d")
    inference_input_path = path_builder.build(
        PathTemplates.model_inference_input_path,
        date=processing_date,
        filename=global_config["datasets"]["inference_input_file_name"],
    )
    assert (
        inference_input_path
        == f"s3://s3_bucket/project_name/business_objective/dataprocessing/inference/model_input/dev/0.0.1-beta/{processing_date}/inference_input_data.parquet"
    )
    # Test read specific env, date, version,
    inference_input_path = path_builder.build(
        PathTemplates.model_inference_input_path,
        filename=global_config["datasets"]["inference_input_file_name"],
        env=env,
        date=date,
        version=version,
    )
    assert (
        inference_input_path
        == "s3://s3_bucket/project_name/business_objective/dataprocessing/inference/model_input/prd/0.0.0/2011-12-12/inference_input_data.parquet"
    )

    training_input_path = path_builder.build(
        PathTemplates.model_training_input_path,
        date=processing_date,
        filename=global_config["datasets"]["training_input_file_name"],
    )

    assert (
        training_input_path
        == f"s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/0.0.1-beta/{processing_date}/model_input_data.parquet"
    )

    training_date = "2024-12-12"
    saved_model_path = path_builder.build(
        PathTemplates.saved_model_path,
        date=training_date,  # always be reading the latest anyway
        filename=global_config["datasets"]["saved_model_file_name"],
    )
    assert (
        saved_model_path
        == "s3://s3_bucket/project_name/business_objective/modelling/trained_model/dev/0.0.1-beta/2024-12-12/saved_model.pickle"
    )

    inference_date = "2024-12-13"
    inference_output_path = path_builder.build(
        PathTemplates.model_inference_output_path,
        date=inference_date,
        filename=global_config["datasets"]["inference_output_file_name"],
    )
    assert (
        inference_output_path
        == "s3://s3_bucket/project_name/business_objective/inference/model_output/dev/0.0.1-beta/2024-12-13/inference_output.parquet"
    )


def test_path_builder_from_yaml_templates():
    env = "prd"  # from user input
    date = "2011-12-12"
    global_config = ConfigYAML.load('tabular_reusable_assets/config/base/globals.yml')

    path_builder = PathBuilder(config=global_config)
    version = "0.0.0"
    # from processing
    processing_date = datetime.now().strftime("%Y-%m-%d")
    inference_input_path = path_builder.build(
        global_config['templates']['model_inference_input_path'],
        date=processing_date,
    )
    assert (
        inference_input_path
        == f"s3://s3_bucket/project_name/business_objective/dataprocessing/inference/model_input/prd/0.0.1-beta/{processing_date}/inference_input_data.parquet"
    )
    # Test read specific env, date, version,
    inference_input_path = path_builder.build(
        global_config['templates']['model_inference_input_path'],
        env=env,
        date=date,
        version=version,
    )
    assert (
        inference_input_path
        == "s3://s3_bucket/project_name/business_objective/dataprocessing/inference/model_input/prd/0.0.0/2011-12-12/inference_input_data.parquet"
    )

    training_input_path = path_builder.build(
        global_config['templates'][ "model_training_input_path" ],
        date=processing_date,
    )

    assert (
        training_input_path
        == f"s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/prd/0.0.1-beta/{processing_date}/model_input_data.parquet"
    )


def test_filepath_partial_format():
    model_training_input_path = FilePath(
        template="s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    )

    model_input_path = model_training_input_path.partial_format(
        s3_bucket="s3_bucket", project_name="project_name", business_objective="business_objective"
    )
    final_path = model_input_path.format(env="env", version="version", date="9999-12-31", filename="hello.py")
    assert (
        final_path
        == "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/env/version/9999-12-31/hello.py"
    )
