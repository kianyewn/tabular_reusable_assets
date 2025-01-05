import pytest
from xgboost import XGBClassifier

from tabular_reusable_assets.model.evaluation_utils import Scorer


@pytest.fixture(scope="session")
def trained_binary_classification_model(titanic_dataset):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]

    default_model = XGBClassifier(random_state=99)
    default_model.fit(
        data[feature_columns],
        data[target_column],
        verbose=False,
    )
    return default_model


def test_scorer_binary_classification(titanic_dataset, trained_binary_classification_model):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]

    scorer = Scorer(["roc_auc", "accuracy", "precision", "recall"])

    metrics = scorer(trained_binary_classification_model, data[feature_columns], data[target_column])
    assert list(metrics.keys()) == ["roc_auc", "accuracy", "precision", "recall"]


def test_scorer_multi_classification(titanic_dataset, trained_binary_classification_model):
    # TODO
    pass


def test_scorer_regression(titanic_dataset, trained_binary_classification_model):
    # TODO
    pass
