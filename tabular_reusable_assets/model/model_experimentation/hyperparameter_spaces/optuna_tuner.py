import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from xgboost import XGBClassifier

from tabular_reusable_assets.model.model_utils import plot_learning_curve

from . import parameter_space_registry
from .base import BaseModelTuner
from .utils import optuna_preview_parameter_space


@dataclass
class TuningConfig:
    # Study configuration
    study_name: str
    direction: str = "maximize"
    n_trials: int = 100
    timeout: Optional[int] = None  # in seconds
    mode: str = "n_trials"  # To tune via 'n_trials' or 'timeout'
    duration: int = 60  # in seconds, ignored if mode is 'n_trials'

    # Model parameters
    metric: str = None
    base_params: Dict[str, Any] = field(default_factory=dict)
    fit_params: Dict[str, Any] = field(default_factory=dict)

    # Evaluation parameters
    scorer_metric: str = "roc_auc"

    # Storage
    study_dir: Optional[str] = None  # Changed to Optional with None default
    storage_type: str = "in_memory"  # Changed default to "in_memory","sqlite"  # or "postgres", "mysql"

    # Other
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    training_verbosity: bool = False

    # Custom parameter space configuration
    param_space_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None

    def __post_init__(self):
        if base_params.get("random_state", None) is None:
            base_params["random_state"] = self.random_state


class ModelHyperParameterTuner(BaseModelTuner):
    """
    Handles hyperparameter optimization for machine learning models using Optuna.

    This class provides automated hyperparameter tuning capabilities for various
    machine learning models (XGBoost, LightGBM, CatBoost, etc.) using Optuna's
    optimization framework.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        tuning_config: TuningConfig,
        # model_master_df: pd.DataFrame,
        group_id_column: str,
        target_column: str,
        weights_column: str,
        base_model: BaseEstimator,
        verbose: bool = False,
        custom_objective: Optional[Callable] = None,
        *input_cols,
        **kwargs,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.tuning_config = tuning_config
        # self.model_master_df = model_master_df
        self.group_id_column = group_id_column
        self.target_column = target_column
        self.weights_column = weights_column
        self.base_model = base_model
        self.verbose = verbose
        self.custom_objective = custom_objective
        self.input_cols = input_cols
        # Setup logging
        self._setup_logging()
        # Initialize study
        self.study = self._create_study()
        self.best_score = float("-inf")
        self.best_model = None
        self.scorer = self.init_scorer()

    def init_scorer(self):
        try:
            if isinstance(self.tuning_config.scorer_metric, str):
                scoring_metric = self.tuning_config.scorer_metric
                scorer = get_scorer(scoring_metric)
            elif isinstance(self.tuning_config.scorer_metric, list):
                # if it is a list, then score is based on first metric defined in list
                scoring_metric = self.tuning_config.scorer_metric[0]
                scorer = get_scorer(scoring_metric)
            self.logger.info(f"Scorer initialized and evaluation will be based on: `{scoring_metric}`")
            return scorer
        except Exception as e:
            self.logger.error(f"Error initializing scorer: {str(e)}")
            raise

    def _setup_logging(self):
        """Setup logging configuration"""
        if self.tuning_config.study_dir is None:
            # If no study_dir, only use StreamHandler
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )
        else:
            log_dir = Path(self.tuning_config.study_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_dir / f"{self.tuning_config.study_name}_{datetime.now():%Y%m%d}.log"),
                    logging.StreamHandler(),
                ],
            )
        self.logger = logging.getLogger(self.tuning_config.study_name)

    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study"""
        storage = None  # Default to in-memory storage

        if self.tuning_config.storage_type == "sqlite":
            if self.tuning_config.study_dir is None:
                raise ValueError("study_dir must be specified when using sqlite storage")
            study_path = Path(self.tuning_config.study_dir)
            study_path.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{study_path}/{self.tuning_config.study_name}.db"
        elif self.tuning_config.storage_type != "in_memory":
            # Add other storage types as needed
            raise ValueError(f"Unsupported storage type: {self.tuning_config.storage_type}")

        return optuna.create_study(
            study_name=self.tuning_config.study_name,
            direction=self.tuning_config.direction,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.tuning_config.random_state),
        )

    def set_parameter_space(self, param_space_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> None:
        """
        Set a custom parameter search space function.

        Args:
            param_space_fn: Function that takes an optuna.Trial and returns parameter space dictionary
        """
        self.tuning_config.param_space_fn = param_space_fn

    def _get_parameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter search space based on model type"""
        if self.tuning_config.param_space_fn is not None:
            # Use custom parameter space if provided
            params = self.tuning_config.param_space_fn(trial)
        else:
            # Fall back to default parameter space
            params = self.tuning_config.optuna_params_space

        # Update with base parameters
        params.update(self.tuning_config.base_params)
        self.logger.info(f"Params: {params}")
        return params

    def _get_model_callbacks(self, trial: optuna.Trial) -> List:
        """Get model-specific callbacks for optimization"""
        callbacks = []

        if isinstance(self.base_model, XGBClassifier):
            callbacks.append(
                optuna.integration.XGBoostPruningCallback(
                    trial, f"validation_1-{self.tuning_config.params.get('eval_metric')}"
                )
            )
        elif hasattr(self.base_model, "__module__") and "lightgbm" in self.base_model.__module__:
            callbacks.append(
                optuna.integration.LightGBMPruningCallback(trial, self.tuning_config.params.get("eval_metric"))
            )

        elif hasattr(self.base_model, "__module__") and "catboost" in self.base_model.__module__:
            callbacks.append(optuna.integration.CatBoostPruningCallback(trial))

        return callbacks

    def objective(self, trial: optuna.Trial) -> float:
        """Default objective function if custom_objective is not provided"""
        try:
            # Get parameters for this trial
            params = self._get_parameter_space(trial)

            # Create and train model
            # Get model-specific callbacks
            callbacks = self._get_model_callbacks(trial)

            model = self.base_model(
                **params,
                n_jobs=self.tuning_config.n_jobs,  # Fixed: changed self.config to self.tuning_config
                random_state=self.tuning_config.random_state,  # Fixed: changed self.config to self.tuning_config
                callbacks=callbacks,
            )

            model.fit(
                self.X_train,
                self.y_train,
                verbose=self.tuning_config.training_verbosity,
                **self.tuning_config.fit_params,
            )
            score = self.scorer(model, self.X_val, self.y_val)
            # Store best iteration
            trial.set_user_attr("best_iteration", model.best_iteration)
            return score  # model.best_score

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def optimize(self) -> Dict[str, Any]:
        """Run optimization"""
        self.logger.info(f"Starting optimization for {self.tuning_config.study_name}")

        try:
            if self.custom_objective:
                objective_func = self.custom_objective
            else:
                objective_func = self.objective

            if self.tuning_config.mode == "n_trials":
                # Run optimization
                self.study.optimize(
                    objective_func,
                    n_trials=self.tuning_config.n_trials,
                    timeout=self.tuning_config.timeout,
                    n_jobs=1,  # Using 1 for better logging
                )
            elif self.tuning_config.mode == "timeout":
                # Run optimization
                # Optuna tune with time based as opposed to n_trials
                sampler = optuna.samplers.TPESampler(seed=self.tuning_config.random_state)
                study = optuna.create_study(direction=self.tuning_config.direction, sampler=sampler)
                tic = time.time()
                while time.time() - tic < self.tuning_config.duration:
                    study.optimize(objective_func, n_trials=1)

            # Save results
            # results = self._save_results()
            self.trial_df = self.study.trials_dataframe()

            self.logger.info(f"Optimization completed. Best value: {self.study.best_value}")
            # return self.trial_df

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

        # Fit with entire best params
        # Get best parameters
        self.best_params = self.study.best_trial.params
        self.best_params.update(self.tuning_config.base_params)

        # Train final model with best parameters, and use best iteration from trial
        base_model_dummy = self.base_model()
        if hasattr(base_model_dummy, "early_stopping_rounds"):
            self.best_params.pop("early_stopping_rounds", None)

        if hasattr(base_model_dummy, "n_estimators"):
            self.best_params.pop("n_estimators", None)
            final_model = self.base_model(
                **self.best_params, n_estimators=self.study.best_trial.user_attrs["best_iteration"]
            )

        else:
            final_model = self.base_model(**self.best_params)

        fit_params = self.tuning_config.fit_params.copy()
        fit_params.pop("eval_set", None)
        final_model.fit(self.X_train, self.y_train, **fit_params)
        self.final_model = final_model
        return self

    def get_learning_curve(self):
        """Retrain the model with the best parameters and return the learning curve"""
        self.logger.info(f"Best params: {self.best_params}")
        audit_model = self.base_model(**self.best_params)
        fit_params = self.tuning_config.fit_params.copy()
        # fit_params.update({"verbose": True})
        audit_model.fit(self.X_train, self.y_train, verbose=self.tuning_config.training_verbosity, **fit_params)
        return audit_model


if __name__ == "__main__":
    base_params = {"eval_metric": "auc", "objective": "binary:logistic"}
    boosting_params = {"early_stopping_rounds": 500, "n_estimators": 10000}

    X_train = pd.read_csv("data/X_train.csv").reset_index()
    X_val = pd.read_csv("data/X_val.csv").reset_index()
    y_train = pd.read_csv("data/y_train.csv").reset_index(drop=True)
    y_val = pd.read_csv("data/y_val.csv").reset_index(drop=True)
    print(y_train["Survived"].value_counts())

    config = TuningConfig(
        study_name="hyper-parameter tuning for xgboost",
        direction="maximize",
        n_trials=10,
        timeout=60 * 30,
        mode="n_trials",
        duration=60 * 30,
        base_params={**base_params, **boosting_params},
        fit_params={"eval_set": [(X_train, y_train), (X_val, y_val)]},
        scorer_metric="roc_auc",
        storage_type="in_memory",
        study_dir=None,  # "trained_models/optuna_studies",
        random_state=42,
        n_jobs=-1,
        verbose=True,
        training_verbosity=False,
        param_space_fn=parameter_space_registry.create_parameter_space_fn("xgboost"),
    )

    experiment = ModelHyperParameterTuner(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        tuning_config=config,
        group_id_column="index",
        target_column="Survived",
        weights_column=None,
        base_model=XGBClassifier,
    )
    # preview parameter space
    if False:
        print(optuna_preview_parameter_space(parameter_space_registry.create_parameter_space_fn("xgboost")))
    experiment.optimize()
    audit_model = experiment.get_learning_curve()
    plot_learning_curve(xgb_model=audit_model, metrics_to_plot=["auc"], legend_labels=["train", "val"])
    print(f"Final validation score: {experiment.scorer(experiment.final_model, X_val, y_val)}")
