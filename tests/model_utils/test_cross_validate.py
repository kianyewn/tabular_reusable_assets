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
    random_search_validation,
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
        return_indices=True,
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
    assert isinstance(cv_results["test_accuracy"], np.ndarray)
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


def test_cross_validate_name_with_only_one_scoring(titanic_dataset, default_xgb_estimator, cross_validator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    cv_results = default_cross_validate(
        estimator=default_xgb_estimator,
        X=data[feature_columns],
        y=data[target_column],
        groups=data[group_column],
        scoring="roc_auc",
        cv=cross_validator,
    )
    assert isinstance(cv_results, dict)
    assert isinstance(cv_results["test_score"], np.ndarray)
    assert "test_accuracy" not in list(cv_results.keys())
    assert list(cv_results.keys()) == [
        "fit_time",
        "score_time",
        "test_score",
        "train_score",
    ]  # by default dont get indices
    assert len(cv_results["fit_time"]) == cross_validator.get_n_splits()
    assert "indices" not in list(cv_results.keys())


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
    estimator = random_search_validation(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=default_xgb_estimator,
        fit_params={"groups": data[group_column]},
        param_distributions=params_dist,
        cv=[(train_idx, test_idx)],  # should be list of iteration
        n_iter=1,
        scoring="accuracy",
        n_jobs=1,  # dont use all cores for testing
        random_state=22,
    )

    # only has one split
    assert "split1_test_score" not in estimator.cv_results_.keys()


def test_3_stage_tuning(titanic_dataset, default_xgb_estimator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]

    n_iter = 3
    stage_1_top_k = 2
    stage_2_top_k = 1
    main_scoring = "roc_auc"
    direction = "maximize"
    greater_is_better = True if direction == "maximize" else False
    sort_ascending = not greater_is_better
    cv = get_default_stratified_group_kfold(n_splits=2)  # for stage 2 and stage 3

    # Create stratified group train test split
    data2, _ = cross_validate_and_create_folds(
        estimator=XGBClassifier(),
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        cv=get_default_stratified_group_kfold(n_splits=2),
        group_column=group_column,
    )
    data2 = data2.sample(frac=0.5, random_state=99)
    train_idx, test_idx = data2[data2["fold"] != 0].index, data2[data2["fold"] == 0].index

    ## Get default score
    ## Default
    default_estimator = random_search_validation(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=XGBClassifier(),
        fit_params={},
        param_distributions={},
        n_iter=1,
        scoring=main_scoring,
        cv=[(train_idx, test_idx)],
        random_state=99,
    )
    default_results = pd.DataFrame(default_estimator.cv_results_)

    # Stage 1 Random search on sampled validation set (Fastest)
    estimator = random_search_validation(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=XGBClassifier(),
        fit_params={},
        param_distributions={"reg_lambda": [0, 1, 2, 3, 4], "reg_alpha": [0, 1, 2, 3, 4]},  # train on sampled dataset
        n_iter=n_iter,
        scoring=main_scoring,
        cv=[(train_idx, test_idx)],
        random_state=99,
    )

    stage_1_results = pd.DataFrame(estimator.cv_results_)
    stage_1_results = pd.concat([default_results, stage_1_results], axis=0, ignore_index=True)
    stage_1_results = (
        stage_1_results.reset_index(names="trial_id")
        .drop("rank_test_score", axis=1)
        .sort_values(by="mean_test_score", ascending=sort_ascending)
    )
    stage_1_results["stage_1_rank_test_score"] = stage_1_results["mean_test_score"].rank(
        method="dense", ascending=sort_ascending
    )
    stage_1_results["pass_stage_1"] = np.where(
        (stage_1_results["stage_1_rank_test_score"] <= stage_1_top_k) | (stage_1_results["trial_id"] == 0), True, False
    )

    passed_top_k_stage_1_results = stage_1_results[stage_1_results["pass_stage_1"]]

    # Stage 2: Evaluate top candidates by 5 fold CV on sampled dataset (Faster)
    stage_2_results_dict = {
        "trial_id": [],
        "stage_2_test_cv_score": [],
        "stage_2_mean_test_cv_score": [],
        "stage_2_std_test_cv_score": [],
    }
    for _, row in passed_top_k_stage_1_results.iterrows():
        params = row["params"]
        cv_results = default_cross_validate(
            estimator=XGBClassifier(random_state=99, **params),
            X=data2[feature_columns],
            y=data2[target_column],
            groups=data2[group_column],
            scoring=(main_scoring,),
            cv=cv,
            return_indices=False,
            return_train_score=False,
        )
        stage_2_results_dict["trial_id"].append(row["trial_id"])
        stage_2_results_dict["stage_2_test_cv_score"].append(cv_results[f"test_{main_scoring}"])
        stage_2_results_dict["stage_2_mean_test_cv_score"].append(np.mean(cv_results[f"test_{main_scoring}"]))
        stage_2_results_dict["stage_2_std_test_cv_score"].append(np.std(cv_results[f"test_{main_scoring}"]))

    stage_2_results = pd.DataFrame(stage_2_results_dict)
    stage_2_results = stage_1_results.merge(stage_2_results, on=["trial_id"], how="left").fillna(
        -np.inf if greater_is_better else np.inf
    )
    stage_2_results["stage_2_rank_test_score"] = stage_2_results["stage_2_mean_test_cv_score"].rank(
        method="dense", ascending=sort_ascending
    )

    stage_2_results["pass_stage_2"] = np.where(
        (stage_2_results["stage_2_rank_test_score"] <= stage_2_top_k) | (stage_2_results["trial_id"] == 0), True, False
    )

    # Stage3: for those records that passed stage 2 (normal)
    passed_top_k_stage_2_results = stage_2_results[stage_2_results["pass_stage_2"]]
    stage_3_results_dict = {
        "trial_id": [],
        "stage_3_test_cv_score": [],
        "stage_3_mean_test_cv_score": [],
        "stage_3_std_test_cv_score": [],
        "stage_3_train_cv_score": [],
        "stage_3_mean_train_cv_score": [],
        "stage_3_std_train_cv_score": [],
    }
    for _, row in passed_top_k_stage_2_results.iterrows():
        params = row["params"]
        cv_results = default_cross_validate(
            estimator=XGBClassifier(random_state=99, **params),
            X=data[feature_columns],
            y=data[target_column],
            groups=data[group_column],
            scoring=(main_scoring,),
            cv=cv,
            return_indices=False,
            return_train_score=True,
        )
        stage_3_results_dict["trial_id"].append(row["trial_id"])
        stage_3_results_dict["stage_3_test_cv_score"].append(cv_results[f"test_{main_scoring}"])
        stage_3_results_dict["stage_3_mean_test_cv_score"].append(np.mean(cv_results[f"test_{main_scoring}"]))
        stage_3_results_dict["stage_3_std_test_cv_score"].append(np.std(cv_results[f"test_{main_scoring}"]))
        stage_3_results_dict["stage_3_train_cv_score"].append(cv_results[f"train_{main_scoring}"])
        stage_3_results_dict["stage_3_mean_train_cv_score"].append(np.mean(cv_results[f"train_{main_scoring}"]))
        stage_3_results_dict["stage_3_std_train_cv_score"].append(np.std(cv_results[f"train_{main_scoring}"]))

    # final results
    stage_3_results = pd.DataFrame(stage_3_results_dict)
    stage_3_results = stage_2_results.merge(stage_3_results, on=["trial_id"], how="left")
    assert list(stage_3_results.columns) == [
        "trial_id",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
        "split0_test_score",
        "mean_test_score",
        "std_test_score",
        "split0_train_score",
        "mean_train_score",
        "std_train_score",
        "param_reg_lambda",
        "param_reg_alpha",
        "stage_1_rank_test_score",
        "pass_stage_1",
        "stage_2_test_cv_score",
        "stage_2_mean_test_cv_score",
        "stage_2_std_test_cv_score",
        "stage_2_rank_test_score",
        "pass_stage_2",
        "stage_3_test_cv_score",
        "stage_3_mean_test_cv_score",
        "stage_3_std_test_cv_score",
        "stage_3_train_cv_score",
        "stage_3_mean_train_cv_score",
        "stage_3_std_train_cv_score",
    ]
