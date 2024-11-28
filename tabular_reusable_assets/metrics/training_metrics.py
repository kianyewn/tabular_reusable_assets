import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tabular_reusable_assets.utils.logger import default_logger as logger

from .average_meter import AverageMeter


@dataclass
class TrainingMetrics:
    enabled: bool = None
    store_history: bool = False
    enabled_metrics: Union[List[str], Set[str]] = field(default_factory=list)

    loss: AverageMeter = field(
        default_factory=lambda: AverageMeter(store_history=True, store_avg_history=True)
    )
    batch_time: Optional[AverageMeter] = None
    data_time: Optional[AverageMeter] = None
    sent_count: Optional[AverageMeter] = None
    scores: Optional[AverageMeter] = None
    grad_norm: Optional[AverageMeter] = None
    learning_rate: Optional[AverageMeter] = None

    _custom_metrics: dict = field(default_factory=list)

    def register_metric(self, name: str, store_avg_history: bool = False):
        """Register a new custom metric.

        Args:
            name: Name of the metric to register
            store_avg_history: Whether to store average history for this metric
        """

        if hasattr(self, name):
            logger.warning(f"Metric: `{name}` already exists as a default metric")

        self._custom_metrics[name] = AverageMeter(
            store_history=self.store_history, store_avg_history=store_avg_history
        )
        if name not in self.enlabled_metrics:
            self.enabled_metrics.append(name)

    def __getattr__(self, name: str) -> Optional[AverageMeter]:
        """Allow access to custom metrics as if they were regular attributes"""
        if name in self._custom_metrics:
            return self._custom_metrics[name]
        raise AttributeError(f"`TrainingMetrics` has no attribute `{name}`")

    def __post_init__(self):
        """Initialize only the enabled metrics"""
        if len(self.enabled_metrics) > 0:
            self.enabled = True

        for metric_name in self.enabled_metrics:
            if hasattr(self, metric_name):
                setattr(
                    self,
                    metric_name,
                    AverageMeter(
                        store_history=self.store_history,
                        store_avg_history=(metric_name == "losses"),
                    ),
                )

            else:
                logger.warning(
                    "Warning, metric: {metric_name} not supported and ignored."
                )

    def __iter__(self):
        for metric_name in self.enabled_metrics:
            if hasattr(self, metric_name):
                yield getattr(self, metric_name)
        # yield custom metrics
        for metric in self._custom_metrics.values():
            yield metric

    def to_dict(self):
        result = {
            metric_name: getattr(self, metric_name)
            for metric_name in self.enabled_metrics
        }
        result.update(self._custom_metrics)
        return result

    def init_before_epoch(self):
        self.start = self.end = time.time()
        self.global_step = 0

    def update_end_time(self):
        if self.enabled:
            self.end = time.time()
            self.global_step += 1

    def update(self, name: str, value: float, n: int = 1):
        """Update metric if enabled, otherwise no-op"""
        if self.enabled and hasattr(self, name) and getattr(self, name) is not None:
            getattr(self, name).update(value, n)

    # def log_step(
    #     self,
    #     step,
    #     epoch,
    #     losses,
    #     lrs,
    #     global_step,
    #     batch_time,
    #     data_time,
    #     sent_count,
    #     scores,
    #     total_steps,
    #     start,
    # ):
    #     # if self.enabled_metrics:
    #     #     logger.info(
    #     #         f"Epoch[{epoch}] "
    #     #         f"steps:[{step +1}/{len(train_dataloader)})] "
    #     #         f"total_elapsed_time: {total_elapsed}, "
    #     #         f"remaining_time: {remaining} "
    #     #         f"data_time: {metrics.data_time.val:.4f} "  # not average to monitor per batch data
    #     #         f"elapsed_batch_time:{metrics.batch_time.val:.4f} "
    #     #         f"sent_count_s: {(metrics.sent_count.avg / metrics.batch_time.avg):.3f} "  # average number of samples per second
    #     #         f"lr: {optimizer.param_groups[0]['lr']:.5f} "
    #     #         f"loss: {metrics.losses.val:.3f} "
    #     #         f"avg loss: {metrics.losses.avg:.3f} "
    #     #         f"score: {metrics.scores.val:.3f} ({metrics.scores.avg})"
    #     #         f"grad: {metrics.grad_values.val:.3f} "
    #     #         f"gradnorm: {grad_norm:.3f} "
    #     #     )
