from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import optuna
from optuna import Trial


@dataclass
class BaseModelHyperParameters:
    """Base class for model-specific hyperparameter definitions"""

    trial: Trial

    def get_params(self) -> Dict[str, Any]:
        """Abstract method to be implemented by each model's parameter class"""
        raise NotImplementedError


class BaseParameterSpaceFunction(Protocol):
    """Protocol defining the interface for parameter space functions"""

    def __call__(self, trial: Trial) -> Dict[str, Any]: ...


class BaseModelTuner(ABC):
    """Base class for model hyperparameter tuning implementations."""

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        """Define the objective function for optimization."""
        pass

    @abstractmethod
    def _get_parameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the parameter search space."""
        pass
