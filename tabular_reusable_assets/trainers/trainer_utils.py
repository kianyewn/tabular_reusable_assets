import inspect
from enum import Enum
from functools import partial


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    EPOCH = "epoch"
    STEPS = "steps"


class SaveStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
    BEST = "best"


class EvaluationStrategy(ExplicitEnum):
    """Deprecated"""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


def number_of_arguments(func):
    if isinstance(func, partial):
        arguments = inspect.signature(func.func).parameters
        return len(arguments) - len(func.args) - len(func.keywords)
    return len(inspect.signature(func).parameters)
