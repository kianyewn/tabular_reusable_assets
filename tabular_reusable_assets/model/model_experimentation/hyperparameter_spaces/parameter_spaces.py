from typing import Any, Dict

from .base import BaseModelHyperParameters
from .registry import HyperParameterSpaceFunctionRegistry


class XGBoostHyperParameters(BaseModelHyperParameters):
    """XGBoost-specific hyperparameter space definition"""

    def get_params(self) -> Dict[str, Any]:
        return {
            "tree_method": self.trial.suggest_categorical("tree_method", ["approx", "hist"]),
            "max_depth": self.trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": self.trial.suggest_int("min_child_weight", 1, 250),
            "subsample": self.trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bynode": self.trial.suggest_float("colsample_bynode", 0.1, 1.0),
            "reg_lambda": self.trial.suggest_float("reg_lambda", 0.001, 25, log=True),
            "reg_alpha": self.trial.suggest_float("reg_alpha", 0.001, 50, log=True),
            "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


class LightGBMHyperParameters(BaseModelHyperParameters):
    """LightGBM-specific hyperparameter space definition"""

    def get_params(self) -> Dict[str, Any]:
        return {
            "num_leaves": self.trial.suggest_int("num_leaves", 20, 3000, log=True),
            "max_depth": self.trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": self.trial.suggest_int("min_child_samples", 1, 250),
            "subsample": self.trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": self.trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_lambda": self.trial.suggest_float("reg_lambda", 0.001, 25, log=True),
            "reg_alpha": self.trial.suggest_float("reg_alpha", 0.001, 50, log=True),
            "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


class CatBoostHyperParameters(BaseModelHyperParameters):
    """CatBoost-specific hyperparameter space definition"""

    def get_params(self) -> Dict[str, Any]:
        return {
            "depth": self.trial.suggest_int("depth", 3, 12),
            "min_child_samples": self.trial.suggest_int("min_child_samples", 1, 250),
            "subsample": self.trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bylevel": self.trial.suggest_float("colsample_bylevel", 0.1, 1.0),
            "reg_lambda": self.trial.suggest_float("reg_lambda", 0.001, 25, log=True),
            "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


# Initialize the registry
parameter_space_registry = HyperParameterSpaceFunctionRegistry()

# Register parameter spaces
parameter_space_registry.register("xgboost", XGBoostHyperParameters)
parameter_space_registry.register("lightgbm", LightGBMHyperParameters)
parameter_space_registry.register("catboost", CatBoostHyperParameters)
