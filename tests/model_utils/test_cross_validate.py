import pytest
from xgboost import XGBClassifier

from tabular_reusable_assets.model.model_utils import default_cross_validate, get_default_stratified_group_kfold


@pytest.fixture(scope="session")
def default_xgb_estimator():
    return XGBClassifier()


def test_cross_validate(titanic_dataset, default_xgb_estimator):
    data = titanic_dataset["data"]
    feature_columns = titanic_dataset["feature_columns"]
    target_column = titanic_dataset["target_column"]
    group_column = titanic_dataset["group_column"]
    cv = get_default_stratified_group_kfold()

    cv_results = default_cross_validate(
        estimator=default_xgb_estimator,
        X=data[feature_columns],
        y=data[target_column],
        scoring=("accuracy", "roc_auc"),
        groups=data[group_column],
        cv=cv,
    )
    assert isinstance(cv_results, dict)

    assert isinstance(cv_results, dict)
    assert list(cv_results.keys()) == ['fit_time', 'score_time', 'indices', 'test_accuracy', 'test_roc_auc']
    assert len(cv_results['fit_time']) == cv.get_n_splits()
    assert len(cv_results['indices']['train']) == len(cv_results['indices']['test'])
