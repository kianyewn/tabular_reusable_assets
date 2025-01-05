import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from tabular_reusable_assets.model.model_utils import (
    create_group_folds,
    cross_validate_and_create_folds,
    default_cross_validate,
    get_default_stratified_group_kfold,
    random_search_cv,
)


@pytest.fixture(scope="session")
def default_xgb_estimator():
    return XGBClassifier()


@pytest.fixture(scope="session")
def cross_validator():
    return get_default_stratified_group_kfold(n_splits=2)


def test_cross_validator_dont_accept_1_split():
    with pytest.raises(
        ValueError,
        match="k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1.",
    ):
        _ = get_default_stratified_group_kfold(n_splits=1)


def test_stratified_group_kfold_consistency(titanic_dataset):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    data_sample = data.sample(20, random_state=22)

    cv = get_default_stratified_group_kfold(n_splits=2)
    test_indices_1 = []
    for _, test_idx in cv.split(
        X=data_sample[feature_columns], y=data_sample[target_column], groups=data_sample["index"]
    ):
        test_indices_1.append(test_idx)
    # split second time
    test_indices_2 = []
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
    assert list(cv_results.keys()) == [
        "fit_time",
        "score_time",
        "indices",
        "test_accuracy",
        "train_accuracy",
        "test_roc_auc",
        "train_roc_auc",
    ]
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
        group_column=group_column,
        cv=cross_validator,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(cv_results, dict)
    assert isinstance(cv_results["test_accuracy"], np.ndarray)
    assert list(cv_results.keys()) == ["fit_time", "score_time", "indices", "test_accuracy", "test_roc_auc"]
    assert len(cv_results["fit_time"]) == cross_validator.get_n_splits()
    assert len(cv_results["indices"]["train"]) == len(cv_results["indices"]["test"])


def test_create_group_folds(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    data_fold_v1, _ = cross_validate_and_create_folds(
        estimator=default_xgb_estimator,
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        scoring=("accuracy", "roc_auc"),
        group_column=group_column,
        cv=cross_validator,
    )

    data_fold_v2 = create_group_folds(
        cv=cross_validator,
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        group_id_column=group_column,
    )

    assert_series_equal(data_fold_v1["fold"], data_fold_v2["fold"])


def test_random_search_cv(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    n_iter = 4
    estimator = random_search_cv(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=default_xgb_estimator,
        fit_params={"groups": data[group_column]},
        param_distributions={"learning_rate": [0.1, 0.2, 0.3, 0.5], "reg_alpha": [1, 2, 3, 4, 5, 6, 7, 8, 10]},
        cv=cross_validator,
        n_iter=n_iter,
        scoring="accuracy",
        n_jobs=1,  # dont use all cores for testing
        random_state=22,
    )

    cv_results = estimator.cv_results_
    assert isinstance(estimator, RandomizedSearchCV)
    assert isinstance(cv_results, dict)

    # Test keys
    keys = [
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "param_reg_alpha",
        "param_learning_rate",
        "params",
        "split0_test_score",
        "split1_test_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "split0_train_score",
        "split1_train_score",
        "mean_train_score",
        "std_train_score",
    ]
    assert list(cv_results.keys()) == keys

    # Test number of rows
    res = pd.DataFrame(estimator.cv_results_)
    assert res.shape[0] == n_iter


def test_random_search_cv_default_params(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    params_dist = {}
    estimator = random_search_cv(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=default_xgb_estimator,
        fit_params={"groups": data[group_column]},
        param_distributions=params_dist,
        cv=cross_validator,
        n_iter=1,
        scoring="accuracy",
        n_jobs=1,  # dont use all cores for testing
        random_state=22,
    )
    estimator_score = estimator.best_score_

    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    # verify against using cross_validate
    cv_results = default_cross_validate(
        estimator=default_xgb_estimator,
        X=data[feature_columns],
        y=data[target_column],
        scoring=("accuracy", "roc_auc"),
        groups=data[group_column],
        cv=cross_validator,
    )
    default_score = np.mean(cv_results["test_accuracy"])
    assert estimator_score == default_score


def test_random_search_validation(titanic_dataset, default_xgb_estimator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=99)

    params_dist = {}
    estimator = random_search_cv(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=default_xgb_estimator,
        fit_params={"groups": data[group_column]},
        param_distributions=params_dist,
        cv=[(train_idx, test_idx)],
        n_iter=1,
        scoring="accuracy",
        n_jobs=1,  # dont use all cores for testing
        random_state=22,
    )

    # only has one split
    assert "split1_test_score" not in estimator.cv_results_.keys()
