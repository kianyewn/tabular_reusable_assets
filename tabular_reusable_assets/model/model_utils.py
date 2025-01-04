import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna import Trial
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_score,
    cross_validate,
)
from xgboost import XGBClassifier


def get_default_stratified_group_kfold(n_splits: int = 5, shuffle: bool = False, random_state: int = None):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def default_cross_validate(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    error_score="raise",
    return_indices=True,
) -> Dict:
    # cross validation score
    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        params=fit_params,
        error_score=error_score,
        return_indices=return_indices,
    )
    return cv_results


def cross_validate_and_create_folds(
    estimator,
    data,
    feature_columns,
    target_column,
    fold_column="fold",
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    error_score="raise",
    return_indices=True,
) -> Dict:
    # cross validation score
    cv_results = cross_validate(
        estimator=estimator,
        X=data[feature_columns],
        y=data[target_column],
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        params=fit_params,
        error_score=error_score,
        return_indices=return_indices,
    )
    test_indices = cv_results["indices"]["test"]
    data[fold_column] = None
    for fold_idx, test_indices in enumerate(test_indices):
        data.loc[test_indices, fold_column] = fold_idx
    return data, cv_results


def create_group_folds(
    cv: BaseCrossValidator,
    data: pd.DataFrame,
    feature_columns: str,
    target_column: str,
    group_id_column: str,
    fold_column: str = "fold",
    verbose: bool = False,
):
    """Creates cross-validation folds and adds them to the input DataFrame.

    Args:
        cv (BaseCrossValidator): A scikit-learn cross-validator instance (e.g., KFold, GroupKFold).
        data (pd.DataFrame): Input DataFrame containing features and target.
        target_column (str): Name of the column containing target values.
        group_id_column (str): Name of the column containing group IDs for grouped cross-validation.
        fold_column (str, optional): Name of the column to store fold indices. Defaults to "fold".

    Returns:
        pd.DataFrame: Original DataFrame with an additional column containing fold indices.

    Example:
        >>> from sklearn.model_selection import GroupKFold
        >>> cv = GroupKFold(n_splits=5)
        >>> df = create_group_folds(cv, data, 'target', 'group_id')
    """
    n_splits = cv.get_n_splits()
    data[fold_column] = None
    for fold, (train_index, val_index) in enumerate(
        cv.split(data[feature_columns], data[target_column], groups=data[group_id_column])
    ):
        if verbose:
            logger.info(
                f"Creating fold {fold+1}/{n_splits} with {len(train_index)} training and {len(val_index)} validation samples"
            )
        data.loc[val_index, fold_column] = fold
    return data


def get_default_random_search_cv(
    estimator: BaseEstimator,
    param_distribution: Union[Dict, List[Dict]],
    n_iter: int = 10,
    scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
    n_jobs: Optional[int] = None,
    refit: Union[bool, str, Callable] = True,
    cv: Optional[Union[int, BaseCrossValidator, Iterable]] = None,
    verbose: int = 0,
    pre_dispatch: str = "2*n_jobs",
    random_state: Optional[int] = None,
    error_score: Union[str, float] = "raise",
    return_train_scores: bool = True,
) -> RandomizedSearchCV:
    estimator = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distribution,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        cv=cv,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        random_state=random_state,
        error_score=error_score,
        return_train_scores=return_train_scores,
    )
    return estimator


def get_default_random_search_cv(
    estimator: BaseEstimator,
    param_distribution: Union[Dict, List[Dict]],
    n_iter: int = 10,
    scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
    n_jobs: Optional[int] = None,
    refit: Union[bool, str, Callable] = True,
    cv: Optional[Union[int, BaseCrossValidator, Iterable]] = None,
    verbose: int = 0,
    pre_dispatch: str = "2*n_jobs",
    random_state: Optional[int] = None,
    error_score: Union[str, float] = "raise",
    return_train_scores: bool = True,
) -> RandomizedSearchCV:
    estimator = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distribution,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        cv=cv,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        random_state=random_state,
        error_score=error_score,
        return_train_scores=return_train_scores,
    )
    return estimator


def get_default_optuna_optimize(
    # data
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: List[str],
    # Model parameters
    base_model: BaseEstimator = None,
    base_params: Dict = None,
    fit_params: Dict = None,
    callbacks: Optional[List[Callable]] = None,
    model_n_jobs: int = -1,
    parameter_search_space: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
    random_state: Optional[int] = None,
    # study parameters
    study_dir: str = None,
    study_name: str = None,
    storage_type: Optional[str] = "in_memory",
    direction: str = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    load_if_exists: Optional[bool] = True,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    study_n_jobs: int = 1,
    # CV parameters
    groups: np.array = None,
    scoring: Union[str, Callable] = None,
    cv: Union[int, BaseCrossValidator, Iterable] = None,
    cv_n_jobs: Optional[int] = None,
    cv_verbose: Optional[int] = 0,
    cv_params: Optional[Dict] = None,
    pre_dispatch: str = "2*n_jobs",
    error_score: Union[str, float] = "raise",
) -> float:
    # Initialize study
    storage = None  # Default to in-memory storage

    if storage_type == "sqlite":
        if study_dir is None:
            raise ValueError("study_dir must be specified when using sqlite storage")
        study_path = Path(study_dir)
        study_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{study_path}/{study_name}.db"

    elif storage_type is None:
        storage = None
    elif storage_type != "in_memory" or storage_type is not None:
        # Add other storage types as needed
        raise ValueError(f"Unsupported storage type: {storage_type}")

    if sampler is not None:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
        )

    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

    parameter_space = None
    if parameter_search_space is None:
        # Parameter space for XGBoost, by default
        def get_parameter_search_space(trial):
            params = {
                "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 250),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 50, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }
            return params

        parameter_space = get_parameter_search_space
    else:
        parameter_space = parameter_search_space

    if cv_params is None:
        cv_params = dict()
    if fit_params is None:
        fit_params = dict()

    def objective_func(trial):
        params = parameter_space(trial)
        params.update(base_params)
        params.pop("n_jobs", None)  # in case base params have n_jobs
        model = base_model(
            **params,
            n_jobs=model_n_jobs,
            callbacks=callbacks,
        )

        cv_params.update(fit_params)

        scores = cross_val_score(
            estimator=model,
            X=data[feature_columns],
            y=data[target_column],
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=cv_n_jobs,
            verbose=cv_verbose,
            params=cv_params,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        return np.mean(scores)

    study.optimize(
        objective_func, n_trials=n_trials, timeout=timeout, n_jobs=study_n_jobs
    )  # Using 1 for better logging
    return study


def cv_prediction(
    data: pd.DataFrame,
    model: BaseEstimator,
    feature_columns: List[str],
    target_column: str = "Survived",
    fold_column: str = "fold",
    scoring: List[str] = ["roc_auc"],
    fit_params: Optional[Dict] = None,
    verbose: bool = False,
    error_handling: str = "raise",
) -> pd.DataFrame:
    """
    Perform cross-validation prediction and scoring.

    Args:
        data: Input DataFrame containing features, target, and fold column
        model: Scikit-learn compatible model
        feature_columns: List of feature column names
        target_column: Name of the target column
        fold_column: Name of the fold column
        scoring: List of scoring metrics to evaluate
        fit_params: Additional parameters to pass to model.fit()
        verbose: Whether to print progress
        error_handling: How to handle errors ('raise' or 'skip')

    Returns:
        DataFrame containing cross-validation results
    """
    cv_results = defaultdict(list)

    try:
        # Validate inputs
        if not all(col in data.columns for col in [*feature_columns, target_column, fold_column]):
            raise ValueError("Missing required columns in input data")

        # Get unique folds
        folds = sorted(data[fold_column].unique())
        total_folds = len(folds)

        for index, fold in enumerate(folds):
            if verbose:
                logger.info(f"Processing fold {index + 1}/{total_folds}")

            try:
                # Split data
                fold_train = data[data[fold_column] != fold]
                fold_val = data[data[fold_column] == fold]

                # Record fold index
                cv_results["cv_fold"].append(index)

                # Fit model
                start_time = time.time()
                fold_model = clone(model)
                fold_model.fit(fold_train[feature_columns], fold_train[target_column], **(fit_params or {}))
                cv_results["fit_time"].append(time.time() - start_time)

                # Score model
                score_start_time = time.time()
                for metric in scoring:
                    try:
                        scorer = get_scorer(metric)
                        cv_results[f"test_{metric}"].append(
                            scorer(model, fold_val[feature_columns], fold_val[target_column])
                        )
                        cv_results[f"train_{metric}"].append(
                            scorer(model, fold_train[feature_columns], fold_train[target_column])
                        )
                    except Exception as e:
                        logger.error(f"Error calculating {metric}: {str(e)}")
                        if error_handling == "raise":
                            raise
                        cv_results[f"test_{metric}"].append(np.nan)
                        cv_results[f"train_{metric}"].append(np.nan)

                cv_results["score_time"].append(time.time() - score_start_time)

            except Exception as e:
                logger.error(f"Error in fold {index}: {str(e)}")
                if error_handling == "raise":
                    raise
                # Add null values for failed fold
                for metric in scoring:
                    cv_results[f"test_{metric}"].append(np.nan)
                    cv_results[f"train_{metric}"].append(np.nan)
                cv_results["fit_time"].append(np.nan)
                cv_results["score_time"].append(np.nan)

        results_df = pd.DataFrame(cv_results)

        # Add summary statistics
        summary_stats = {
            "model": model,
            "mean": results_df.mean(),
            "std": results_df.std(),
            "min": results_df.min(),
            "max": results_df.max(),
        }

        return results_df, summary_stats

    except Exception as e:
        logger.error(f"Fatal error in cv_prediction: {str(e)}")
        raise


def two_stage_cv_prediction(
    data: pd.DataFrame,
    model: BaseEstimator,
    feature_columns: List[str],
    target_column: str = "Survived",
    fold_column: str = "fold",
    scoring: List[str] = ["roc_auc"],
    fit_params: Optional[Dict] = None,
    verbose: bool = False,
    error_handling: str = "raise",
    early_stopping_strategy: str = "median",  # 'median', 'mean', 'max', or 'percentile'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform cross-validation prediction with proper early stopping handling.
    # Two stage CV:
    Key considerations:
    Consistency: Using a fixed number of iterations for final evaluation ensures consistency across folds
    Validation: The two-stage approach helps validate that the chosen number of iterations generalizes well
    Efficiency: You can parallelize the final CV since you're not using early stopping
    Robustness: Using median/percentile instead of mean helps handle outlier folds
    Memory: This approach is more memory-efficient since you don't need to store multiple early-stopping models
    The two-stage approach is generally considered the most robust, as it:
    Finds a stable number of iterations
    Prevents overfitting to any single fold
    Provides more reliable cross-validation scores
    Makes the final model more interpretable

    Args:
        data: Input DataFrame containing features, target, and fold column
        model: Scikit-learn compatible model
        feature_columns: List of feature column names
        target_column: Name of the target column
        fold_column: Name of the fold column
        scoring: List of scoring metrics to evaluate
        fit_params: Additional parameters to pass to model.fit()
        verbose: Whether to print progress
        error_handling: How to handle errors ('raise' or 'skip')
        early_stopping_strategy: How to aggregate best_iterations across folds

    Returns:
        DataFrame containing cross-validation results
    """
    cv_results = defaultdict(list)
    best_iterations = []

    # Stage 1: Get best_iterations from each fold
    for fold in sorted(data[fold_column].unique()):
        fold_train = data[data[fold_column] != fold]
        fold_val = data[data[fold_column] == fold]

        # Fit with early stopping
        model_fold = clone(model)  # Create fresh model instance
        model_fold.fit(
            fold_train[feature_columns],
            fold_train[target_column],
            eval_set=[(fold_val[feature_columns], fold_val[target_column])],
            **(fit_params or {}),
            verbose=verbose,
        )

        if hasattr(model_fold, "best_iteration"):
            best_iterations.append(model_fold.best_iteration)

    # Determine final n_estimators
    if early_stopping_strategy == "median":
        final_n_estimators = int(np.median(best_iterations))
    elif early_stopping_strategy == "mean":
        final_n_estimators = int(np.mean(best_iterations))
    elif early_stopping_strategy == "max":
        final_n_estimators = max(best_iterations)
    elif early_stopping_strategy == "percentile":
        final_n_estimators = int(np.percentile(best_iterations, 75))

    # Stage 2: Run final CV with fixed n_estimators
    final_params = model.get_params()
    final_params["n_estimators"] = final_n_estimators
    final_params.pop("early_stopping_rounds", None)

    # Create new model with final parameters
    final_model = clone(model).__class__().set_params(**final_params)

    # Run final CV scoring
    try:
        # Validate inputs
        if not all(col in data.columns for col in [*feature_columns, target_column, fold_column]):
            raise ValueError("Missing required columns in input data")

        # Get unique folds
        folds = sorted(data[fold_column].unique())
        total_folds = len(folds)

        for index, fold in enumerate(folds):
            if verbose:
                logger.info(f"Processing fold {index + 1}/{total_folds}")

            try:
                # Split data
                fold_train = data[data[fold_column] != fold]
                fold_val = data[data[fold_column] == fold]

                final_model = clone(final_model).set_params(**final_params)
                final_model.fit(
                    fold_train[feature_columns], fold_train[target_column], **(fit_params or {}), verbose=verbose
                )
                # Record fold index
                cv_results["cv_fold"].append(index)

                # Score model
                score_start_time = time.time()
                for metric in scoring:
                    try:
                        scorer = get_scorer(metric)
                        cv_results[f"test_{metric}"].append(
                            scorer(final_model, fold_val[feature_columns], fold_val[target_column])
                        )
                        cv_results[f"train_{metric}"].append(
                            scorer(final_model, fold_train[feature_columns], fold_train[target_column])
                        )
                    except Exception as e:
                        logger.error(f"Error calculating {metric}: {str(e)}")
                        if error_handling == "raise":
                            raise
                        cv_results[f"test_{metric}"].append(np.nan)
                        cv_results[f"train_{metric}"].append(np.nan)

                cv_results["score_time"].append(time.time() - score_start_time)

            except Exception as e:
                logger.error(f"Error in fold {index}: {str(e)}")
                if error_handling == "raise":
                    raise
                # Add null values for failed fold
                for metric in scoring:
                    cv_results[f"test_{metric}"].append(np.nan)
                    cv_results[f"train_{metric}"].append(np.nan)
                cv_results["fit_time"].append(np.nan)
                cv_results["score_time"].append(np.nan)

        results_df = pd.DataFrame(cv_results)

        # Add summary statistics
        summary_stats = {
            "model": final_model,
            "best_iteration": final_n_estimators,
            "mean": results_df.mean(),
            "std": results_df.std(),
            "min": results_df.min(),
            "max": results_df.max(),
        }

        return results_df, summary_stats

    except Exception as e:
        logger.error(f"Fatal error in cv_prediction: {str(e)}")
        raise


def plot_learning_curve(
    xgb_model: XGBClassifier, metrics_to_plot: List[str] = ["mlogloss"], legend_labels: List[str] = ["train", "val"]
):
    """Plots the learning curve for an XGBoost model showing training and validation metrics.

    Args:
        xgb_model (XGBClassifier): A trained XGBoost classifier with evaluation results.
        metrics_to_plot (List[str], optional): List of metrics to plot. Defaults to ["mlogloss"].
        legend_labels (List[str], optional): Labels for the training and validation curves.
            Defaults to ["train", "val"].

    Raises:
        AttributeError: If eval_set was not specified during model training.

    Example:
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        >>> plot_learning_curve(model, metrics_to_plot=["mlogloss", "merror"])
    """
    if not xgb_model.evals_result_:
        raise AttributeError("You did not specify `eval_set` in .fit()")

    evals_result = xgb_model.evals_result_
    for idx, (eval_dtype, result) in enumerate(evals_result.items()):
        for metric in result:
            if metric in metrics_to_plot:
                metric_history = result[metric]
                x = np.arange(len(metric_history))
                plt.plot(x, metric_history, label=f"{legend_labels[idx]}_{metric}")

    plt.legend(loc="right")
    plt.show()
    return
