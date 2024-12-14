from typing import Dict, List
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from tabular_reusable_assets.datasets import ExampleDatasets

# class BasicModelUtils:
#     def stratified_grouped_train_test_split(X, y, test_size=0.2, random_state=42):
#         pass

#     def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
#         pass

#     def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
#         pass


class XGBModel:
    def __init__(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        target: str,
        group: List[str] = None,
        params: Dict = None,
        tree_params: Dict = None,
        fit_params: Dict = None,
        cv_params: Dict = None,
        random_state: int = None,
    ):
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.target = target

        if params is None:
            params = {}
        if tree_params is None:
            tree_params = {}
        if fit_params is None:
            fit_params = {}

        self.params = params
        self.fixed_params = params
        self.fixed_params.update(tree_params)
        self.fit_params = fit_params
        self.cv_params = cv_params

        if params.get("random_state", None) is None and random_state is not None:
            self.params.update({"random_state": random_state})

        if group is not None and not isinstance(group, list):
            group = [group]

        self.group = group

        self.model = None
        self.random_state = self.fixed_params.get("random_state", None)
        self.cv_score = None
        self.cross_val_predictions = None

    def fit(self, X, y, **fit_params):
        if fit_params:
            self.fit_params.update(fit_params)

        self.model = XGBClassifier(**self.fixed_params)
        self.model.fit(X, y, **self.fit_params)
        return

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def predict_proba(self, X):
        y_pred_proba = self.model.predict_proba(X)
        return y_pred_proba

    def single_metric_cross_val_score(self):
        """cross_val_score can only take 1 scoring, while cross_validate can take multiple scoring"""
        # cross validation score
        cv_score = cross_val_score(
            estimator=self.model,
            X=self.dataset.loc[:, self.feature_columns],
            y=self.dataset.loc[:, self.target],
            groups=self.dataset.loc[:, self.group].values.reshape(-1)
            if self.group is not None
            else None,
            scoring="accuracy",
            cv=StratifiedGroupKFold(
                n_splits=self.cv_params.get("n_splits", 5),
                shuffle=True,
                random_state=self.fixed_params.get("random_state", self.random_state),
            ),
            n_jobs=None,
            verbose=0,
            params=self.fit_params,
            error_score=np.nan,
        )
        return cv_score

    def train(self):
        self.model = XGBClassifier(**self.fixed_params)
        self.model.fit(
            self.dataset[self.feature_columns],
            self.dataset[self.target],
            **self.fit_params,
        )

        # cross validation score
        cv_score = cross_validate(
            estimator=self.model,
            X=self.dataset.loc[:, self.feature_columns],
            y=self.dataset.loc[:, self.target],
            groups=self.dataset.loc[:, self.group].values.reshape(-1)
            if self.group is not None
            else None,
            scoring=("accuracy", "roc_auc"),
            cv=StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            n_jobs=None,
            verbose=0,
            params=self.fit_params,
            error_score=np.nan,
            return_indices=True,
        )

        fold_indices = cv_score["indices"]["test"]
        self.dataset["fold"] = None
        for fold_idx, fold_indices in enumerate(fold_indices):
            self.dataset.loc[fold_indices, "fold"] = fold_idx
        cv_score.pop("indices")  # remove from memory
        self.cv_score = cv_score

        # cross validation metrics over entire dataset
        cv_predictions = cross_val_predict(
            estimator=self.model,
            X=self.dataset.loc[:, self.feature_columns],
            y=self.dataset.loc[:, self.target],
            groups=self.dataset.loc[:, self.group].values.reshape(-1)
            if self.group is not None
            else None,
            cv=StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            n_jobs=None,
            verbose=0,
            params=self.fit_params,
            method="predict_proba",
        )  # (cv,method.shape[1])

        self.cross_val_predictions = cv_predictions

        self.dataset["cv_predicted_prob_class=0_({0})".format(self.target)] = (
            cv_predictions[:, 0]
        )
        self.dataset["cv_predicted_prob_class=1_({0})".format(self.target)] = (
            cv_predictions[:, 1]
        )
        self.dataset["cv_predicted_class_({0})".format(self.target)] = (
            cv_predictions[:, 1] > self.cv_params.get('predicted_class_threshold', 0.5)
        ).astype(int)

        y_true = self.dataset.loc[:, self.target]
        auc = roc_auc_score(y_true, cv_predictions[:, 1])

        cm = confusion_matrix(
            y_true,
            self.dataset["cv_predicted_class_({0})".format(self.target)],
            labels=[0, 1],
        )
        print(cm)

        ## TODO: add other metrics, and then save it as final pd.DataFrame table
        return self.dataset

    @staticmethod
    def plot_learning_curve(
        xgb_model, metrics_to_plot=["mlogloss"], labels=["train", "val"]
    ):
        if not xgb_model.evals_result_:
            raise AttributeError("You did not specify `eval_set` in .fit()")

        evals_result = xgb_model.evals_result_
        for idx, (eval_dtype, result) in enumerate(evals_result.items()):
            for metric in result:
                if metric in metrics_to_plot:
                    metric_history = result[metric]
                    x = np.arange(len(metric_history))
                    plt.plot(x, metric_history, label=f"{labels[idx]}_{metric}")

        plt.legend(loc="right")
        plt.show()

        
    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        # - Load the model from disk
        instance = joblib.load(path)
        return instance 



if __name__ == "__main__":
    data_output = ExampleDatasets.load_binary_classification_breast_cancer_dataset()
    data = data_output.data
    numerical_features = data_output.numerical_features
    categorical_features = data_output.categorical_features
    feature_columns = numerical_features + categorical_features

    target = data_output.target

    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data.loc[:, target])
    xgb_model = XGBModel(
        dataset=data,
        feature_columns=feature_columns,
        target=target,
        group=numerical_features[0],
        random_state=42,
        cv_params={"n_splits": 5, 'predicted_class_threshold':0.5},
        params={"eval_metric": "auc"},
        fit_params={"eval_set": [(train.loc[:, feature_columns], train.loc[:, target])], 'verbose':False}
    )
    xgb_model.fit(X=data.loc[:, feature_columns], y=data.loc[:, target])
    y_pred_proba = xgb_model.predict_proba(X=data.loc[:, feature_columns])
    # print(y_pred_proba.shape)
    scored_dataset = xgb_model.train()
    # print(scored_dataset)
    # xgb_model.save('xgb_model.joblib')

    # xgb_model = XGBModel.load('xgb_model.joblib')
    # print(xgb_model.cv_score,  xgb_model.model.__sklearn_is_fitted__())

    # xgb_base = XGBClassifier(objective="binary:logistic", eval_metric="auc")
    # XGBModel.plot_learning_curve(xgb_base, metrics_to_plot=["auc", "mlogloss"])
