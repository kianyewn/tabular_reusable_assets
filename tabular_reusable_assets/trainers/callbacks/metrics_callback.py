import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tabular_reusable_assets.metrics.training_metrics import TrainingMetrics
from tabular_reusable_assets.trainers.callbacks.base_callback import TrainerCallback
from tabular_reusable_assets.utils.logger import default_logger as logger
from tabular_reusable_assets.utils.utils import timeStat


class MetricsCallback(TrainerCallback):
    def __init__(
        self,
        metrics_to_track: List[str],
        log_dir: Path,
        experiment_name: str,
        store_history: bool = False,
    ):
        self.store_history = store_history
        # self.metrics = {}
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name

        # for metric_name in metrics_to_track:
        #     self.metrics[metric_name] = AverageMeter(
        #         store_history=store_history, store_avg_history=(metric_name == "losses")
        #     )
        self.metrics = TrainingMetrics(
            enabled_metrics=metrics_to_track, store_history=store_history
        )

        # storeage for all metircs
        self.batch_metrics_history = []
        self.epoch_metrics_history = []

        # create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # define log file paths
        self.batch_log_path = self.log_dir / "batch_metrics.csv"
        self.epoch_log_path = self.log_dir / "epoch_metrics.csv"

        
        # Initialize buffer settings
        self.buffer_size = 100  # Configure how many rows to store before writing
        self._metrics_buffer = []  # Initialize empty buffer
        

    def on_training_start(self):
        """Called at the start of training"""
        self.training_start_time = time.time()
        # Reset log files
        for path in [self.batch_log_path, self.epoch_log_path]:
            if path.exists():
                logger.info(f"Removing file: {path}")
                path.unlink()

    def on_epoch_start(self):
        self.epoch_start_time = time.time()
        self.prev_batch_end_time = time.time()

    def on_epoch_end(self):
        self._flush_metrics_buffer()

    def on_batch_start(self, batch: int, logs: dict = None):
        for name, value in logs.items():
            if name in self.metrics:
                value = value.pop("value", None) if isinstance(value, dict) else value
                n = value.pop("n", 1) if isinstance(value, dict) else 1
                # Update metrics name
                self.metrics.update(name, val=value, n=n)
            return

    def on_batch_end(self, batch: int, logs: dict = None):
        # Initialize metric row
        batch_metrics_row = {
            "epoch": logs.get("epoch", 0),
            "batch": batch,
            "timestamp": datetime.now(),
            "experiment_name": self.experiment_name,
        }

        metrics_dict = self.metrics.to_dict()
        for name, value in logs.items():
            if name in metrics_dict:
                n = value.pop("n", 1) if isinstance(value, dict) else 1
                val = value.pop("value", None) if isinstance(value, dict) else value
                self.metrics.update(name, value=val, n=n)

                batch_metrics_row[name] = (
                    f"{metrics_dict[name].val}({metrics_dict[name].avg})"
                )

        # Log at specific intervals
        total_steps = logs.get("total_steps", None)
        epoch = logs.get("epoch", 0)

        if (
            (batch == 0)
            or ((batch + 1) % self.print_freq == 0)
            or (batch + 1 == total_steps)
        ):
            # Calculate timing
            total_elapsed, remaining = timeStat(
                start=self.epoch_start_time, percent=(batch + 1) / total_steps
            )

            # Build log message
            metrics_dict = self.metrics.to_dict()
            log_msg = (
                f"Epoch[{epoch}] "
                f"Step:[{batch + 1}/{total_steps}] "
                f"Time:[{total_elapsed}<{remaining}] "
            )

            # Add available metrics to log message
            if metrics_dict.get("data_time"):
                log_msg += f"data_time:{metrics_dict['data_time'].val:.4f} "
            if metrics_dict.get("batch_time"):
                log_msg += f"batch_time:{metrics_dict['batch_time'].val:.4f} "
            if metrics_dict.get("sent_count") and metrics_dict.get("batch_time"):
                samples_per_sec = (
                    metrics_dict["sent_count"].avg / metrics_dict["batch_time"].avg
                )
                log_msg += f"samples/sec:{samples_per_sec:.3f} "
            if metrics_dict.get("losses"):
                log_msg += f"loss:{metrics_dict['losses'].val:.3f}({metrics_dict['losses'].avg:.3f}) "
            if metrics_dict.get("scores"):
                log_msg += f"score:{metrics_dict['scores'].val:.3f}({metrics_dict['scores'].avg:.3f}) "
            if metrics_dict.get("grad_values"):
                log_msg += f"grad:{metrics_dict['grad_values'].val:.3f} "
            if metrics_dict.get("lrs"):
                log_msg += f"lr:{metrics_dict['lrs'].val:.5f} "

            logger.info(log_msg)

        # Add to buffer and flush if needed
        self._metrics_buffer.append(batch_metrics_row)
        
        if len(self._metrics_buffer) >= self.buffer_size:
            self._flush_metrics_buffer()
        return

    def _flush_metrics_buffer(self) -> None:
        """Write all buffered metrics to disk and clear buffer."""
        if not self._metrics_buffer:  # Skip if buffer is empty
            return
            
        try:
            batch_metric_df = pd.DataFrame(self._metrics_buffer)
            if self.batch_log_path.exists():
                batch_metric_df.to_csv(
                    self.batch_log_path,
                    mode="a",
                    header=False,
                    index=False
                )
            else:
                batch_metric_df.to_csv(
                    self.batch_log_path,
                    index=False
                )
            self._metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer: {str(e)}")

    def on_training_end(self):
        """Ensure any remaining metrics are written when training ends."""
        self._flush_metrics_buffer()

    def get_current_avg_metrics(self):
        """Return current averaged metrics"""
        return {name: metric.avg for name, metric in self.metrics}

    def get_current_metrics(self):
        return {name: metric.val for name, metric in self.metrics}
