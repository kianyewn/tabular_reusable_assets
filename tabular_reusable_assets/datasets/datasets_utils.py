from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable, Union


class KedroPathTemplates:
    raw_data = "s3://{s3_bucket}/{project_name}/{business_objective}/01_raw_data/{env}/{version}/{processing_date}/{raw_data_filename}"
    intermediate_data = "s3://{s3_bucket}/{project_name}/{business_objective}/02_intermediate_data/{env}/{version}/{processing_date}/{intermediate_data_filename}"
    processed_data = "s3://{s3_bucket}/{project_name}/{business_objective}/03_processed_data/{env}/{version}/{processing_date}/{processed_data_filename}"
    features_data = "s3://{s3_bucket}/{project_name}/{business_objective}/04_features_data/{env}/{version}/{processing_date}/{features_data_filename}"
    model_input_data = "s3://{s3_bucket}/{project_name}/{business_objective}/05_model_input_data/{env}/{version}/{processing_date}/{model_input_data_filename}"
    models = "s3://{s3_bucket}/{project_name}/{business_objective}/06_models/{env}/{version}/{processing_date}/{models_data_filename}"
    model_output_data = "s3://{s3_bucket}/{project_name}/{business_objective}/07_model_output_data/{env}/{version}/{processing_date}/{model_output_data_filename}"
    reporting_data = "s3://{s3_bucket}/{project_name}/{business_objective}/08_reporting_data/{env}/{version}/{processing_date}/{reporting_data_filename}"

    def arguments(
        self,
        s3_bucket,
        project_name,
        business_objective,
        env,
        version,
        processing_date,
        raw_data_filename=None,
        intermediate_data_filename=None,
        processed_data_filename=None,
        features_data_filename=None,
        model_input_data_filename=None,
        models_data_filename=None,
        model_output_data_filename=None,
        reporting_data_filename=None,
    ):
        return {
            "s3_bucket": s3_bucket,
            "project_name": project_name,
            "business_objective": business_objective,
            "env": env,
            "version": version,
            "processing_date": processing_date,
            "raw_data_filename": raw_data_filename,
        }


class KedroPathTemplatesCategory:
    raw_data = "s3://{s3_bucket}/{project_name}/{business_objective}/01_raw_data/{category}/{env}/{version}/{processing_date}/{raw_data_filename}"
    intermediate_data = "s3://{s3_bucket}/{project_name}/{business_objective}/02_intermediate_data/{category}/{env}/{version}/{processing_date}/{intermediate_data_filename}"
    processed_data = "s3://{s3_bucket}/{project_name}/{business_objective}/03_processed_data/{category}/{env}/{version}/{processing_date}/{processed_data_filename}"
    features_data = "s3://{s3_bucket}/{project_name}/{business_objective}/04_features_data/{category}/{env}/{version}/{processing_date}/{features_data_filename}"
    model_input_data = "s3://{s3_bucket}/{project_name}/{business_objective}/05_model_input_data/{category}/{env}/{version}/{processing_date}/{model_input_data_filename}"
    models = "s3://{s3_bucket}/{project_name}/{business_objective}/06_models/{category}/{env}/{version}/{processing_date}/{models_data_filename}"
    model_output_data = "s3://{s3_bucket}/{project_name}/{business_objective}/07_model_output_data/{category}/{env}/{version}/{processing_date}/{model_output_data_filename}"
    reporting_data = "s3://{s3_bucket}/{project_name}/{business_objective}/08_reporting_data/{category}/{env}/{version}/{processing_date}/{reporting_data_filename}"


@dataclass
class FilePath:
    filepath: str = None
    template: Union[str, Callable] = None

    def format(self, s3_bucket, project_name, business_objective, env, version, date, filename):
        if self.template is None:
            raise ValueError("{self.__class__.name__}'s `template` is `None`. Please provide a string template.")

        return self.template.format(
            s3_bucket=s3_bucket,
            project_name=project_name,
            business_objective=business_objective,
            env=env,
            version=version,
            date=date,
            filename=filename,
        )

    def partial_format(self, **kwargs):
        setattr(self, "format", partial(self.template.format, **kwargs))
        return self


class PathTemplates:
    model_training_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/training/model_input/{env}/{version}/{date}/{filename}"
    model_inference_input_path = "s3://{s3_bucket}/{project_name}/{business_objective}/dataprocessing/inference/model_input/{env}/{version}/{date}/{filename}"
    saved_model_path = "s3://{s3_bucket}/{project_name}/{business_objective}/modelling/trained_model/{env}/{version}/{date}/{filename}"
    model_inference_output_path = (
        "s3://{s3_bucket}/{project_name}/{business_objective}/inference/model_output/{env}/{version}/{date}/{filename}"
    )

    def get_model_training_input(self, env, version, processing_date, model_training_input_filename):
        return self.model_training_input_path.format(
            env=env,
            version=version,
            date=processing_date,
            model_training_input_filename=model_training_input_filename,
        )

    def get_model_inference_input(self, env, version, processing_date, model_inference_input_filename):
        return self.model_inference_input_path.format(
            env=env,
            version=version,
            date=processing_date,
            model_inference_input_filename=model_inference_input_filename,
        )

    def get_saved_model(self, env, version, training_date, trained_model_filename):
        return self.saved_model_path.format(
            env=env, version=version, date=training_date, trained_model_filename=trained_model_filename
        )

    def get_model_inference_output(self, env, version, inference_date, model_inference_output_filename):
        return self.model_inference_output_path.format(
            env=env,
            version=version,
            date=inference_date,
            model_inference_output_filename=model_inference_output_filename,
        )


class PathBuilder:
    def __init__(self, config: dict, templates=None):
        self.config = config
        self.s3_bucket = config["s3_bucket"]
        self.project_name = config["project_name"]
        self.business_objective = config["business_objective"]
        self.env = config["env"]
        self.path_templates = PathTemplates() if templates is None else templates

    def build(
        self,
        path_template: str,
        filename: str=None,
        env: str = None,
        date: str = None,
        version: str = None,
        **kwargs
    ):
        """Builds the path for the model training input.

        Args:
            path_template (str): The path template to use.
            date (_type_): The date to use.
            filename (_type_): The filename to use.
            version (_type_, optional): The version to use. Defaults to None.

        Returns:
            str: The path for the model training input.
        """
        env = self.config["env"] if env is None else env
        date = self.config["date"] if date is None else date
        version = self.config["version"] if version is None else version
        return path_template.format(
            s3_bucket=self.s3_bucket,
            project_name=self.project_name,
            business_objective=self.business_objective,
            env=env,
            version=version,
            date=date,
            filename=filename,
            **kwargs
        )

    def build_path(self, template_key: str, **kwargs):
        path_template = getattr(self, template_key)
        return path_template.format(**kwargs)


if __name__ == "__main__":
    env = "dev"  # from user input
    date = datetime.now().strftime("%Y-%m-%d")
    global_config = {
        "s3_bucket": "s3_bucket",
        "project_name": "project_name",
        "business_objective": "business_objective",
        "version": "0.0.1-beta",
        "env": "dev",
        "datasets": {
            "training_input_file_name": "model_input_data.parquet",
            "inference_input_file_name": "infernence_input_data.parquet",
            "saved_model_file_name": "saved_model.pickle",
            "inference_output_file_name": "inference_output.parquet",
        },
    }
    path_builder = PathBuilder(config=global_config)
    version = "0.0.1-beta"
    # from processing
    processing_date = datetime.now().strftime("%Y-%m-%d")
    inference_input_path = path_builder.build(
        PathTemplates.model_inference_input_path,
        date=processing_date,
        filename=global_config["datasets"]["inference_input_file_name"],
    )
    print(inference_input_path)
    training_input_path = path_builder.build(
        PathTemplates.model_training_input_path,
        date=processing_date,
        filename=global_config["datasets"]["training_input_file_name"],
    )

    print(inference_input_path)

    training_date = "2024-12-12"
    saved_model_path = path_builder.build(
        PathTemplates.saved_model_path,
        date=training_date,
        filename=global_config["datasets"]["saved_model_file_name"],
    )
    print(saved_model_path)

    inference_date = "2024-12-13"
    inference_output_path = path_builder.build(
        PathTemplates.model_inference_output_path,
        date=inference_date,
        filename=global_config["datasets"]["inference_output_file_name"],
    )
    print(inference_output_path)
