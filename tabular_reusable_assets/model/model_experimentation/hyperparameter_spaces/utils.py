from typing import Any, Dict

import optuna

from .base import BaseParameterSpaceFunction


def optuna_preview_parameter_space(param_space_fn: BaseParameterSpaceFunction) -> Dict[str, Any]:
    """
    Preview the parameter space by generating one set of parameters for Optuna

    Args:
        param_space_fn: Function that defines the parameter space

    Returns:
        Dict containing one set of sampled parameters
    """
    study = optuna.create_study()
    trial = study.ask()
    params = param_space_fn(trial)
    study.tell(trial, 0.0)
    return params
