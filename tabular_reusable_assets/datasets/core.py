import typing as T
from abc import ABC, abstractmethod

from pydantic import BaseModel

from tabular_reusable_assets.utils.file_helper import FileHelper


class Reader(ABC, BaseModel, strict=True, frozen=True, extra="forbid"):
    KIND: str
    limit: int | None = None
    _DEFAULT_READ_ARGS: dict[str, T.Any] = {}

    @abstractmethod
    def read(self):
        pass

    def try_get_latest_file(self):
        if "9999-12-31" in self.path:
            return FileHelper.get_latest_file(self.path)
        return self.path


class Writer(ABC, BaseModel, strict=True, frozen=True, extra="forbid"):
    KIND: str
    _DEFAULT_WRITE_ARGS: dict[str, T.Any] = {}

    @abstractmethod
    def write(self):
        pass
