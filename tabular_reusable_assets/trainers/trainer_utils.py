from enum import Enum

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
    NO = 'no'
    EPOCH = 'epoch'
    STEPS = 'steps'

class SaveStrategy(ExplicitEnum):
    NO = 'no'
    STEPS = 'steps'
    EPOCH = 'epoch'
    BEST = 'best'

class EvaluationStrategy(ExplicitEnum):
    NO = 'no'
    STEPS = 'steps'
    EPOCH = 'epoch'