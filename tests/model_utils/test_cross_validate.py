import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from tabular_reusable_assets.model.model_utils import (
    cross_validate_and_create_folds,
    default_cross_validate,
    get_default_stratified_group_kfold,
)


@pytest.fixture(scope="session")
def default_xgb_estimator():
    return XGBClassifier()


@pytest.fixture(scope="session")
def cross_validator():
    return get_default_stratified_group_kfold()


def test_stratified_group_kfold_consistency(titanic_dataset):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    data_sample = data.sample(20, random_state=22)

    cv = get_default_stratified_group_kfold(n_splits=2)
    test_indices_1 = []
    test_indices_2 = []
    for _, test_idx in cv.split(
        X=data_sample[feature_columns], y=data_sample[target_column], groups=data_sample["index"]
    ):
        test_indices_1.append(test_idx)
    for _, test_idx in cv.split(
        X=data_sample[feature_columns], y=data_sample[target_column], groups=data_sample["index"]
    ):
        test_indices_2.append(test_idx)
    assert np.allclose(test_indices_2, test_indices_1)


def test_cross_validate(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    cv_results = default_cross_validate(
        estimator=default_xgb_estimator,
        X=data[feature_columns],
        y=data[target_column],
        scoring=("accuracy", "roc_auc"),
        groups=data[group_column],
        cv=cross_validator,
    )

    assert isinstance(cv_results, dict)
    assert list(cv_results.keys()) == ["fit_time", "score_time", "indices", "test_accuracy", "test_roc_auc"]
    assert len(cv_results["fit_time"]) == cross_validator.get_n_splits()
    assert len(cv_results["indices"]["train"]) == len(cv_results["indices"]["test"])


def test_cross_validate_create_folds(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    data, cv_results = cross_validate_and_create_folds(
        estimator=default_xgb_estimator,
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        scoring=("accuracy", "roc_auc"),
        groups=data[group_column],
        cv=cross_validator,
    )
    assert isinstance(data, pd.DataFrame)

    assert isinstance(cv_results, dict)
    assert list(cv_results.keys()) == ["fit_time", "score_time", "indices", "test_accuracy", "test_roc_auc"]
    assert len(cv_results["fit_time"]) == cross_validator.get_n_splits()
    assert len(cv_results["indices"]["train"]) == len(cv_results["indices"]["test"])
