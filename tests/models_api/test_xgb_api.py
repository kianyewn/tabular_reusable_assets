import pytest
from xgboost import XGBClassifier


@pytest.fixture(scope="session")
def train(titanic_dataset):
    data = titanic_dataset["data"]
    return data.sample(frac=0.5).reset_index(drop=True)


@pytest.fixture(scope="session")
def val(titanic_dataset):
    data = titanic_dataset["data"]
    return data.sample(frac=0.5).reset_index(drop=True)


def test_xgb_best_score_api(titanic_dataset, train):
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]

    default_model = XGBClassifier(random_state=99)
    default_model.fit(train[feature_columns], train[target_column])

    with pytest.raises(AttributeError, match="`best_score` is only defined when early stopping is used."):
        _ = default_model.best_score


def test_default_xgb_param(titanic_dataset, train, val):
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]

    default_model = XGBClassifier(random_state=99, early_stopping_rounds=5)
    default_model.fit(
        train[feature_columns],
        train[target_column],
        eval_set=[(val[feature_columns], val[target_column])],
        verbose=False,
    )
    default_best_score = default_model.best_score

    explicit_default_model = XGBClassifier(
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_lambda=1,
        reg_alpha=0,
        tree_method="auto",
        scale_pos_weight=1,
        random_state=99,
        early_stopping_rounds=5,
    )

    explicit_default_model.fit(
        train[feature_columns],
        train[target_column],
        eval_set=[(val[feature_columns], val[target_column])],
        verbose=False,
    )
    explicit_default_best_score = explicit_default_model.best_score

    assert default_best_score == explicit_default_best_score


def test_eval_set_api(titanic_dataset, train, val):
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    model = XGBClassifier()

    model.fit(
        train[feature_columns],
        train[target_column],
        eval_set=[(val[feature_columns], val[target_column])],
        verbose=False,
    )
    assert isinstance(model.evals_result_, dict)
    assert list(model.evals_result_.keys()) == ["validation_0"]
    # default metric is logloss
    assert list(model.evals_result_["validation_0"].keys()) == ["logloss"]
    assert len(model.evals_result_["validation_0"]["logloss"]) == 100  # default n_estimators

    model.fit(
        train[feature_columns],
        train[target_column],
        eval_set=[(train[feature_columns], train[target_column]), (val[feature_columns], val[target_column])],
        verbose=False,
    )
    assert list(model.evals_result_.keys()) == ["validation_0", "validation_1"]


def test_eval_metric_eval_set_api(titanic_dataset, train, val):
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]

    model = XGBClassifier(eval_metric=["logloss", "auc"])  # last metric will be used for ES

    with pytest.raises(AttributeError, match="'XGBClassifier' object has no attribute 'evals_result_'"):
        # train without eval_set
        model.fit(train[feature_columns], train[target_column], verbose=False)
        model.evals_result_

    # train with eval_set
    model = XGBClassifier(eval_metric=["logloss", "auc"])
    model.fit(
        train[feature_columns],
        train[target_column],
        eval_set=[(train[feature_columns], train[target_column]), (val[feature_columns], val[target_column])],
        verbose=False,
    )

    assert list(model.evals_result_["validation_0"].keys()) == ["logloss", "auc"]
    assert list(model.evals_result_["validation_1"].keys()) == ["logloss", "auc"]
