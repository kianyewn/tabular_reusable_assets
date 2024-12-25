from abc import ABC, abstractmethod
from typing import Dict

import yaml
from loguru import logger

from .file_helper import FileHelper, PathParser


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


class ConfigYAML(Reader, Writer):
    @staticmethod
    def load(yaml_path: str):
        with open(yaml_path, "r") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        logger.info(f"Successfully loaded file from `{yaml_path}`.")
        return config

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
