from dataclasses import dataclass, field
from typing import Union, Optional

from tabular_reusable_assets.trainers.trainer_utils import (
    EvaluationStrategy,
    IntervalStrategy,
    SaveStrategy,
)

@dataclass
class TrainingArguments:
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps", metadata="The logging strategy to use."
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="steps", metadata={"help": "The checkpoint save strategy to use."}
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no", metadata={"help": "The evaluation strategy to use."}
    )
    evaluation_strategy: Union[EvaluationStrategy, str] = field(
        default=None, metadata={"help": "Deprecated. Use `eval_strategy` instead"}
    )  
    
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )

    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

    def __post_init__(self):
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)