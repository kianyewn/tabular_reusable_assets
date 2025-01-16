import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from loguru import logger

from .file_helper import FileHelper


class Reader(ABC):
    @abstractmethod
    def load(self, file_location):
        pass


class Writer(ABC):
    @abstractmethod
    def save(self, file_location):
        pass


# Custom constructor for !from_file
def from_file_constructor(loader, node):
    value = loader.construct_scalar(node)  # Read the tag's value
    file_path, key = value.split(":", 1)  # Separate file path and key

    # Load the external YAML file
    with open(file_path, "r") as f:
        external_data = yaml.load(f, Loader=yaml.FullLoader)

    # Return the requested key's value
    return external_data[key]


# https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml, gs: yaml python variables concat string
## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


# Add the custom constructor to the YAML loader
# yaml.add_constructor("!from_file", from_file_constructor)
yaml.add_constructor("!from_file", from_file_constructor, Loader=yaml.FullLoader)
## register the tag handler
yaml.add_constructor("!join", join)


@dataclass
class ConfigYAML(Reader, Writer):
    path: str
    config: Optional[Dict[str, Union[str, Dict]]] = None

    def __post_init__(self):
        self.config = self.load(self.path)
        for key, value in self.config.items():
            setattr(self, key, value)

    @staticmethod
    def load(yaml_path: str):
        with open(yaml_path, "r") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        logger.info(f"Successfully loaded file from `{yaml_path}`.")
        return config

    def __getitem__(self, key: str):
        if hasattr(self, key):
            print("has attr")
            return getattr(self, key)
        else:
            raise KeyError(f"Key: {key} does not exist in config")

    def keys(self):
        return self.config.keys()

    def items(self):
        return self.config.items()

    def values(self):
        return self.config.values()

    def __len__(self):
        return len(self.config)

    @staticmethod
    def save(obj: Dict, yaml_path: str):
        with open(yaml_path, "w") as stream:
            yaml.dump(obj, stream)
        logger.info(f"Successfully saved file to  `{yaml_path}`.")
        return

    @staticmethod
    def delete(yaml_path: str) -> None:
        FileHelper.remove_file(yaml_path)
        logger.info(f"Successfully deleted file from `{yaml_path}`.")


class ConfigClass:
    def __init__(self, config: Dict):
        for key, value in config.items():
            setattr(self, key, value)

    def dict(self):
        return self.__dict__


class ConfigHelper:
    def __init__(self, base_config_path: str):
        self.config_cache = {}
        self.base_config = self.load_yaml(base_config_path)
        self.global_vars = {
            "ENV": os.getenv("ENV", "dev"),
            "S3_BUCKET": os.getenv("S3_BUCKET", "default-bucket"),
            "PROJECT_NAME": os.getenv("PROJECT_NAME", "default-project"),
            "VERSION": os.getenv("VERSION", "v1"),
            "DATE_STR": os.getenv("DATE_STR", "latest"),
        }

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def resolve_templates(
        self, config_dict: Dict[str, Any], variables: Dict[str, str], hist_resolved_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively resolve template strings in a dictionary"""
        resolved_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and "{" in value:
                template = value  # Template(value)
                try:
                    try:
                        resolved_dict[key] = template.format(**variables)  # template.substitute(variables)
                    except Exception:
                        all_current_vars = variables.copy()
                        all_current_vars.update(hist_resolved_dict)
                        resolved_dict[key] = template.format(**all_current_vars)

                except KeyError as e:
                    raise ValueError(f"Could not resolve variable {e} in template {value}")
                    print(f"Warning: Could not resolve variable {e} in template {value}")
                    resolved_dict[key] = value
            elif isinstance(value, dict):
                resolved_dict[key] = self.resolve_templates(value, variables, hist_resolved_dict)
            else:
                resolved_dict[key] = value
            if isinstance(value, str):
                # only save in memory if the value is a string
                hist_resolved_dict.update(resolved_dict)
        return resolved_dict

    def get_config(self, config_path: str) -> Dict[str, Any]:
        """Load and resolve a config file with template variables"""
        if config_path not in self.config_cache:
            config = self.load_yaml(config_path)
            # Merge global variables with any local variables in the config
            variables = {**self.global_vars, **config.get("variables", {})}
            resolved_config = self.resolve_templates(config, variables, hist_resolved_dict={})
            self.config_cache[config_path] = resolved_config

        return self.config_cache[config_path]

    def update_global_vars(self, **kwargs):
        """Update global variables (e.g., from command line arguments)"""
        self.global_vars.update(kwargs)
        # Clear cache since variables changed
        self.config_cache.clear()
