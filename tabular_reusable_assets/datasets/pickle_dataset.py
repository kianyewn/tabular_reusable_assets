import typing as T

import joblib
from pydantic import Field

from tabular_reusable_assets.datasets.core import Reader, Writer


class PickleDataset(Reader, Writer):
    KIND: T.Literal["PickleDataset"] = "PickleDataset"
    path: str = Field(..., frozen=False)
    read_args: T.Dict[str, T.Any] = None
    write_args: T.Dict[str, T.Any] = None
    template: str = None
    _DEFAULT_READ_ARGS: T.Dict[str, T.Any] = {}
    _DEFAULT_WRITE_ARGS: T.Dict[str, T.Any] = {}

    def model_post_init(self, _context: T.Any = None) -> None:
        self._read_args = {**self._DEFAULT_READ_ARGS, **(self.read_args or {})}
        self._write_args = {**self._DEFAULT_WRITE_ARGS, **(self.write_args or {})}

    def read(self, **read_args) -> T.Dict[str, T.Any]:
        self._read_args.update(read_args)
        return joblib.load(self.path, **self._read_args)

    def write(self, data: T.Dict[str, T.Any], **write_args):
        self._write_args.update(write_args)
        return joblib.dump(data, self.path, **self._write_args)
