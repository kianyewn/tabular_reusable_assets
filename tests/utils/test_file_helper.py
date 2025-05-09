from pathlib import Path
from unittest.mock import patch

from tabular_reusable_assets.utils.file_helper import FileHelper


def test_get_dir_before_date():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = FileHelper.get_dir_before_date(model_training_input_path.replace("{date}", "2022-12-31"))
    assert (
        dir_before_date
        == "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/"
    )


def test_get_dir_before_date_with_regex():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = FileHelper.get_dir_before_date(
        model_training_input_path.replace("{date}", "2022-12-31"), date_regex=r"\d{4}-\d{2}-\d{2}"
    )
    assert (
        dir_before_date
        == "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/"
    )


def test_get_dir_before_date_with_custom_regex():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = FileHelper.get_dir_before_date(model_training_input_path, date_regex=r"{date}")
    assert (
        dir_before_date
        == "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/"
    )


def test_get_latest_file():
    files = [
        "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/2021-01-18/model_input_data.parquet",
        "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/2022-01-18/model_input_data.parquet",
        "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/lolthisthing/model_input_data.parquet",
        "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/2025-01-18/model_input_data.parquet",
    ]

    latest_file = FileHelper.get_latest_file_from_filepaths(files, date_regex=r".*(\d{4}-\d{2}-\d{2}).*")
    assert (
        latest_file
        == "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/2025-01-18/model_input_data.parquet"
    )


def test_replace_date_in_path():
    # Test with replacement date
    path = "data/2025-01-01/train.csv"
    assert FileHelper.replace_date_in_path(path, "2024-12-31") == "data/2024-12-31/train.csv"

    # Test with no replacement (should return original)
    assert FileHelper.replace_date_in_path(path) == path

    # Test with no date in path
    path_no_date = "data/train.csv"
    assert FileHelper.replace_date_in_path(path_no_date, "2024-12-31") == path_no_date

    # Test with multiple dates
    path_multiple = "data/2025-01-01/2024-12-31/train.csv"
    assert FileHelper.replace_date_in_path(path_multiple, "2023-06-30") == "data/2023-06-30/2023-06-30/train.csv"


@patch("tabular_reusable_assets.utils.file_helper.Path")
def test_glob_files(mock_path):
    # Setup mock paths to be returned
    mock_paths = [Path("data/file1.csv"), Path("data/file2.csv")]
    mock_path.return_value.rglob.return_value = mock_paths
    files = FileHelper.glob_files("data", "*.csv")
    assert len(list(files)) == 2


@patch("tabular_reusable_assets.utils.file_helper.joblib.load")
def test_load_dict(mock_joblib):
    dummy_expected = {"hello": "world"}
    mock_joblib.return_value = dummy_expected

    return_dummy = FileHelper.load_dict("Helo World")
    assert return_dummy == dummy_expected
    mock_joblib.assert_called_once_with("Helo World")


@patch("tabular_reusable_assets.utils.file_helper.joblib.dump")
def test_save_dict(mock_joblib):
    dummy = {"hello": "world"}
    FileHelper.save_dict(dummy, "Hello World")
    mock_joblib.assert_called_once_with(dummy, "Hello World")
