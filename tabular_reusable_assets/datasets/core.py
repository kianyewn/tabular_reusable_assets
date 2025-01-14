import typing as T
from abc import ABC, abstractmethod

from pydantic import BaseModel


class Reader(ABC, BaseModel, strict=True, frozen=True, extra="forbid"):
    KIND: str
    limit: int | None = None
    _DEFAULT_READ_ARGS: dict[str, T.Any] = {}

    @abstractmethod
    def read(self):
        pass


class Writer(ABC, BaseModel, strict=True, frozen=True, extra="forbid"):
    KIND: str
    _DEFAULT_WRITE_ARGS: dict[str, T.Any] = {}

    @abstractmethod
    def write(self):
        pass
