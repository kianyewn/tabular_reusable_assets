from typing import Optional

import numpy as np

from tabular_reusable_assets.trainers.trainer_utils import IntervalStrategy
from tabular_reusable_assets.utils.logger import default_logger as logger

from .base_callback import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: Optional[float] = 0,
    ):
        """
        A [`TrainerCallback`] that handles early stopping.

        Args:
            early_stopping_patience (`int`):
                Use with `metric_for_best_model` to stop training when the specified metric worsens for
                `early_stopping_patience` evaluation calls.
            early_stopping_threshold(`float`, *optional*):
                Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
                specified metric must improve to satisfy early stopping conditions. `

        This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
        in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
        early stopping will not occur until the next save step.
        """

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        

    def check_metric_value(self, args, state, control, metric_value):
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_training_start(self, args, state, control, **kwargs):
        assert (
            args.load_best_model_at_end
        ), "EarlyStoppingCallback requires load_best_model_at_end=True"

        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"

        assert (
            args.eval_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model

        if hasattr(metrics, metric_to_check):
            metric = getattr(metrics, metric_to_check)
        else:
            raise ValueError(f"Metric {metric_to_check} not found in metrics")
            
        metric_value = metric.avg

        # if not metric_to_check.startswith("eval_"):
            # metric_to_check = f"eval_{metric_to_check}"
        # metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }
