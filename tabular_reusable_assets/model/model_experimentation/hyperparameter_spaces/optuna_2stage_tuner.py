import os


print("Current working directory:", os.getcwd())
print("Script location:", os.path.abspath(__file__))

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from xgboost import XGBClassifier

from tabular_reusable_assets.model.model_utils import plot_learning_curve

from . import parameter_space_registry
from .base import BaseModelTuner
from .utils import preview_parameter_space


class TuningStage(Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"


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
    stage_1_fixed_learning_rate: float = 0.8
    stage_2_fixed_learning_rate: float = 0.01
    stage_1_n_trials: int = 50
    stage_2_n_trials: int = 50
    current_stage: TuningStage = TuningStage.STAGE_1

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

    def set_stage_1_learning_rate(self, learning_rate: float):
        self.stage_1_fixed_learning_rate = learning_rate

    def set_stage_2_learning_rate(self, learning_rate: float):
        self.stage_2_fixed_learning_rate = learning_rate

    def get_stage_params(self) -> Dict[str, Any]:
        """Get parameters specific to current tuning stage"""
        if self.current_stage == TuningStage.STAGE_1:
            return {"learning_rate": self.stage_1_fixed_learning_rate, "n_trials": self.stage_1_n_trials}
        return {"learning_rate": self.stage_2_fixed_learning_rate, "n_trials": self.stage_2_n_trials}


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
        group_id_column: str = None,
        target_column: str = None,
        weights_column: str = None,
        base_model: BaseEstimator = None,
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

    def init_scorer(self):
        return get_scorer(self.tuning_config.metric)

    def set_stage_1_learning_rate(self, learning_rate: float):
        self.tuning_config.set_stage_1_learning_rate(learning_rate)

    def set_stage_2_learning_rate(self, learning_rate: float):
        self.tuning_config.set_stage_2_learning_rate(learning_rate)

    def stage_1_fixed_learning_rate_exploration(self, learning_rate: float = 0.8):
        """Explore the effect of a fixed learning rate on the model's performance"""
        base_params = self.tuning_config.base_params
        base_params["learning_rate"] = learning_rate
        model = self.base_model(**base_params)
        start_time = time.time()
        model.fit(
            self.X_train, self.y_train, **self.tuning_config.fit_params, verbose=self.tuning_config.training_verbosity
        )
        score = self.scorer(model, self.X_val, self.y_val)
        end_time = time.time()
        self.logger.info(
            f"Time taken to train model: {(end_time - start_time):.2f} seconds, best_iteration: {model.best_iteration}, score: {score:2f}"
        )
        return model

    def stage_2_fixed_learning_rate_exploration(self, learning_rate: float = 0.8):
        """Explore the effect of a fixed learning rate on the model's performance"""
        stage_1_best_params = self.stage_1_best_params
        stage_1_best_params["learning_rate"] = learning_rate
        model = self.base_model(**stage_1_best_params)
        start_time = time.time()
        logger.info(f"Tuning Config: {self.tuning_config.fit_params}")
        logger.info(f"Tuning Config: {stage_1_best_params}")
        model.fit(
            self.X_train, self.y_train, **self.tuning_config.fit_params, verbose=self.tuning_config.training_verbosity
        )
        score = self.scorer(model, self.X_val, self.y_val)
        end_time = time.time()
        self.logger.info(
            f"Time taken to train model: {(end_time - start_time):.2f} seconds, best_iteration: {model.best_iteration}, score: {score:2f}"
        )
        return model

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
        """Get parameter space based on current stage"""
        """Define parameter search space based on model type"""
        if self.tuning_config.param_space_fn is not None:
            # Use custom parameter space if provided
            params = self.tuning_config.param_space_fn(trial)
        else:
            # Fall back to default parameter space
            params = self.tuning_config.optuna_params_space

        # Update with base parameters
        params.update(self.tuning_config.base_params)
        # self.logger.info(f"Params: {params}")

        # Override learning rate based on stage
        stage_params = self.tuning_config.get_stage_params()
        params["learning_rate"] = stage_params["learning_rate"]
        # logger.info(f"params: {params}")
        return params

    def _get_model_callbacks(self, trial: optuna.Trial) -> List:
        """Get model-specific callbacks for optimization"""
        callbacks = []

        if isinstance(self.base_model, XGBClassifier):
            callbacks.append(
                optuna.integration.XGBoostPruningCallback(trial, f"validation_1-{self.tuning_config.metric}")
            )
        elif hasattr(self.base_model, "__module__") and "lightgbm" in self.base_model.__module__:
            callbacks.append(optuna.integration.LightGBMPruningCallback(trial, self.tuning_config.metric))

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
            # TODO: Create a function to do the scoring
            # Store best iteration
            trial.set_user_attr("best_iteration", model.best_iteration)
            return model.best_score

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def optimize(self) -> "ModelHyperParameterTuner":
        """Run two-stage optimization"""
        self.logger.info("Starting two-stage hyperparameter optimization")

        try:
            # Stage 1: Optimize tree parameters with high learning rate
            self.logger.info("Starting Stage 1 optimization")
            stage_1_results = self._run_stage(TuningStage.STAGE_1)

            # Stage 2: Fine-tune with low learning rate
            self.logger.info("Starting Stage 2 optimization")
            stage_2_results = self._run_stage(TuningStage.STAGE_2)

            # Store final results
            self.stage_1_results = stage_1_results
            self.stage_2_results = stage_2_results

            return self

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

    def _run_stage(self, stage: TuningStage) -> Dict[str, Any]:
        """Run optimization for a specific stage"""
        # Update current stage
        self.tuning_config.current_stage = stage
        stage_params = self.tuning_config.get_stage_params()
        # Create new study for this stage
        study = optuna.create_study(
            study_name=f"{self.tuning_config.study_name}_{stage.value}",
            direction=self.tuning_config.direction,
            sampler=optuna.samplers.TPESampler(seed=self.tuning_config.random_state),
        )

        # Run optimization
        study.optimize(self.objective, n_trials=stage_params["n_trials"], n_jobs=1)

        # Get best parameters and model
        best_params = study.best_trial.params
        best_params.update(self.tuning_config.base_params)
        best_iteration = study.best_trial.user_attrs.get("best_iteration")

        # Train final model for this stage
        self.logger.info(f"Best params: {best_params}")
        final_model = self._train_final_model(best_params, best_iteration)

        if stage == TuningStage.STAGE_1:
            self.stage_1_best_params = best_params
        return {
            "study": study,
            "best_params": best_params,
            "best_score": study.best_value,
            "best_model": final_model,
            "best_iteration": best_iteration,
            "trials_dataframe": study.trials_dataframe(),
        }

    def _train_final_model(self, params: Dict[str, Any], best_iteration: Optional[int]) -> BaseEstimator:
        """Train final model with best parameters"""
        # Remove early stopping if present
        params = params.copy()
        params.pop("early_stopping_rounds", None)
        params.pop("n_estimators", None)

        # Create and train model
        if best_iteration:
            model = self.base_model(**params, n_estimators=best_iteration)
        else:
            model = self.base_model(**params)

        # Train without evaluation set
        fit_params = self.tuning_config.fit_params.copy()
        fit_params.pop("eval_set", None)

        combined_X = pd.concat([self.X_train, self.X_val])
        combined_y = pd.concat([self.y_train, self.y_val])
        model.fit(combined_X, combined_y, **fit_params)
        return model

    def get_best_model(self, stage: Optional[TuningStage] = None) -> BaseEstimator:
        """Get best model from specified stage or stage 2 by default"""
        if stage == TuningStage.STAGE_1:
            return self.stage_1_results["best_model"]
        return self.stage_2_results["best_model"]

    def get_best_params(self, stage: Optional[TuningStage] = None) -> Dict[str, Any]:
        """Get best parameters from specified stage or stage 2 by default"""
        if stage == TuningStage.STAGE_1:
            return self.stage_1_results["best_params"]
        return self.stage_2_results["best_params"]

    def plot_optimization_history(self, stage: Optional[TuningStage] = None):
        """Plot optimization history for specified stage"""
        import matplotlib.pyplot as plt

        if stage:
            results = self.stage_1_results if stage == TuningStage.STAGE_1 else self.stage_2_results
            title = f"Optimization History - {stage.value}"
        else:
            # Plot both stages
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Stage 1
            df1 = self.stage_1_results["trials_dataframe"]
            ax1.plot(df1["number"], df1["value"])
            ax1.set_title("Stage 1 Optimization")

            # Stage 2
            df2 = self.stage_2_results["trials_dataframe"]
            ax2.plot(df2["number"], df2["value"])
            ax2.set_title("Stage 2 Optimization")

            plt.tight_layout()
            return fig


if __name__ == "__main__":
    base_params = {"eval_metric": "auc", "objective": "binary:logistic"}
    boosting_params = {"early_stopping_rounds": 500, "n_estimators": 10000}

    X_train = pd.read_csv("data/X_train.csv").reset_index()
    X_val = pd.read_csv("data/X_val.csv").reset_index()
    y_train = pd.read_csv("data/y_train.csv").reset_index(drop=True)
    y_val = pd.read_csv("data/y_val.csv").reset_index(drop=True)
    print(y_train["Survived"].value_counts())

    config = TuningConfig(
        study_name="xgboost_two_stage_tuning",
        direction="maximize",
        stage_1_fixed_learning_rate=0.8,
        stage_2_fixed_learning_rate=0.01,
        stage_1_n_trials=10,
        stage_2_n_trials=10,
        metric="roc_auc",
        base_params={**base_params, **boosting_params},
        fit_params={"eval_set": [(X_train, y_train), (X_val, y_val)]},
        training_verbosity=False,
        random_state=42,
        param_space_fn=parameter_space_registry.create_parameter_space_fn("xgboost"),
    )

    tuner = ModelHyperParameterTuner(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        tuning_config=config,
        base_model=XGBClassifier,
    )
    # Experiment to find a learning rate that that takes a few seconds (but performance is still okay)
    tuner.stage_1_fixed_learning_rate_exploration(learning_rate=0.001)
    tuner.set_stage_1_learning_rate(0.001)
    stage_1_results = tuner._run_stage(TuningStage.STAGE_1)
    tuner.stage_2_fixed_learning_rate_exploration(learning_rate=0.0001)
    stage_2_results = tuner._run_stage(TuningStage.STAGE_2)

    # tuner.optimize()

    # stage1_model = tuner.get_best_model(TuningStage.STAGE_1)
    # stage2_model = tuner.get_best_model(TuningStage.STAGE_2)

    tuner.plot_optimization_history()

    # stage1_curves = tuner.get_learning_curve(TuningStage.STAGE_1)
    # stage2_curves = tuner.get_learning_curve(TuningStage.STAGE_2)
