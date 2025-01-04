from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def project_root():
    """Fixture providing the project root path"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Fixure providing data directory path"""
    return DATA_DIR


@pytest.fixture(scope="session")
def titanic_dataset(data_dir):
    return {
        "data": pd.read_csv(data_dir / "titanic_dataset_sample.csv"),
        "feature_columns": ["Embarked", "Parch", "SibSp", "Sex", "Pclass", "Ticket", "Cabin", "Age", "Fare"],
        "target_column": "Survived",
        "group_column": "index",
    }
