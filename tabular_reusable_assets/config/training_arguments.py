from dataclasses import dataclass, field
from typing import Optional, Union

from tabular_reusable_assets.trainers.trainer_utils import (
    EvaluationStrategy,
    IntervalStrategy,
    SaveStrategy,
)
from tabular_reusable_assets.utils.logger import default_logger as logger


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )

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
        default=None,
        metadata={"help": "The metric to use to compare two different models."},
    )
    greater_is_better: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether the `metric_for_best_model` should be maximized or not."
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": (
                "Log very x update steps. if `logging_strategy='steps'`. Should be an integer. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    save_steps: int = field(
        default=10,
        metadata={
            "help": (
                'Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a'
                "float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    eval_steps: int = field(
        default=10,
        metadata={
            "help": (
                'Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same value as `logging_steps` if not set.'
                "Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )

    max_steps: int = field(
    default=-1,
    metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})

    def __post_init__(self):
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)

        self.save_strategy = SaveStrategy(self.save_strategy)

        # If use eval_strategy with IntervalStrategy.STEPS, and eval step not specified, use logging steps
        if self.eval_strategy == IntervalStrategy.STEPS and (
            self.eval_steps == 0 or self.eval_steps is None
        ):
            if self.logging_steps > 0:
                logger.info(
                    "using `logging_steps` to initialize `eval_steps` to {self.logging_steps}"
                )
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero --eval_steps or --logging_steps"
                )
        # if self.load_best_model_at_end, check if logging steps and evaluation steps are multiples of each other (if they are compatible)
        if self.load_best_model_at_end:
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"
                )
            # if save step is not a multiple of evaluation steps, then we cannot get the best model at the end, because no model is evaluated
            # E.g. save_step = 2 and eval_step=10. We cannot save at step 2 because we do not know what is the best model at step 10.
            if self.save_strategy == SaveStrategy.STEPS and (
                self.save_steps % self.eval_steps != 0
            ):
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                    "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                    f"{self.save_steps} and eval_steps {self.eval_steps}."
                )
