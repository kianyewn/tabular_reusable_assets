from .parameter_spaces import CatBoostHyperParameters, LightGBMHyperParameters, XGBoostHyperParameters
from .registry import parameter_space_registry
from .utils import preview_parameter_space


# Register parameter spaces
parameter_space_registry.register("xgboost", XGBoostHyperParameters)
parameter_space_registry.register("lightgbm", LightGBMHyperParameters)
parameter_space_registry.register("catboost", CatBoostHyperParameters)

__all__ = [
    "XGBoostHyperParameters",
    "LightGBMHyperParameters",
    "CatBoostHyperParameters",
    "parameter_space_registry",
    "preview_parameter_space",
]
