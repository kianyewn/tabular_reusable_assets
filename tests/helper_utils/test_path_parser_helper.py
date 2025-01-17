from tabular_reusable_assets.utils.file_helper import PathParser


def test_get_dir_before_date():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = PathParser.get_dir_before_date(model_training_input_path.replace("{date}", "2022-12-31"))
    assert (
        dir_before_date
        == "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/"
    )


def test_get_dir_before_date_with_regex():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = PathParser.get_dir_before_date(
        model_training_input_path.replace("{date}", "2022-12-31"), date_regex=r"\d{4}-\d{2}-\d{2}"
    )
    assert (
        dir_before_date
        == "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/"
    )


def test_get_dir_before_date_with_custom_regex():
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    dir_before_date = PathParser.get_dir_before_date(model_training_input_path, date_regex=r"{date}")
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

    latest_file = PathParser.get_latest_file(files, date_regex=r".*(\d{4}-\d{2}-\d{2}).*")
    assert (
        latest_file
        == "s3://s3_bucket/project_name/business_objective/dataprocessing/training/model_input/dev/v0/2025-01-18/model_input_data.parquet"
    )
