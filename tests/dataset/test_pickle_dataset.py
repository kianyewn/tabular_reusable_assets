import pytest

from tabular_reusable_assets.datasets.pickle_dataset import PickleDataset


@pytest.fixture
def pickle_dataset():
    return PickleDataset(path="test.pkl")


def test_read(mocker, pickle_dataset):
    mock_load = mocker.patch("joblib.load", return_value={"key": "value"})

    result = pickle_dataset.read()

    mock_load.assert_called_once_with("test.pkl", **{})
    assert result == {"key": "value"}


def test_read_with_args(mocker, pickle_dataset):
    mock_load = mocker.patch("joblib.load", return_value={"key": "value"})

    result = pickle_dataset.read(mmap_mode="r")

    mock_load.assert_called_once_with("test.pkl", mmap_mode="r")
    assert pickle_dataset._read_args == {"mmap_mode": "r"}


def test_write(mocker, pickle_dataset):
    mock_dump = mocker.patch("joblib.dump", return_value=None)
    data = {"key": "value"}

    pickle_dataset.write(data)

    mock_dump.assert_called_once_with(data, "test.pkl", **{})


def test_write_with_args(mocker, pickle_dataset):
    mock_dump = mocker.patch("joblib.dump", return_value=None)
    data = {"key": "value"}

    pickle_dataset.write(data, compress=3)

    mock_dump.assert_called_once_with(data, "test.pkl", compress=3)
    assert pickle_dataset._write_args == {"compress": 3}
