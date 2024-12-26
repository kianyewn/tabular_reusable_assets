from typing import Any, Dict, Type

from optuna import Trial

from .base import BaseModelHyperParameters, BaseParameterSpaceFunction


class HyperParameterSpaceRegistry:
    """Registry for managing hyperparameter space definitions"""

    def __init__(self):
        self._parameter_spaces: Dict[str, Type[BaseModelHyperParameters]] = {}

    def register(self, model_name: str, param_space_class: Type[BaseModelHyperParameters]) -> None:
        """
        Register a new parameter space class for a model type.

        Args:
            model_name: Name identifier for the model
            param_space_class: Class defining the parameter space
        """
        self._parameter_spaces[model_name] = param_space_class

    def get_parameter_space(self, model_name: str) -> Type[BaseModelHyperParameters]:
        """
        Retrieve the parameter space class for a given model type.

        Args:
            model_name: Name identifier for the model

        Returns:
            Parameter space class for the specified model

        Raises:
            KeyError: If model_name is not registered
        """
        if model_name not in self._parameter_spaces:
            raise KeyError(f"No parameter space registered for model: {model_name}")
        return self._parameter_spaces[model_name]

    def create_parameter_space_fn(self, model_name: str) -> BaseParameterSpaceFunction:
        """
        Create a parameter space function for a given model type.

        Args:
            model_name: Name identifier for the model

        Returns:
            Function that generates parameter space for the specified model
        """
        param_space_class = self.get_parameter_space(model_name)

        def param_space_fn(trial: Trial) -> Dict[str, Any]:
            return param_space_class(trial).get_params()

        return param_space_fn


parameter_space_registry = HyperParameterSpaceRegistry()
