import json
import os
import sys
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from termcolor import colored
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from tabular_reusable_assets.utils.logger import default_logger as logger

from .callbacks.base_callback import TrainerCallback
from .callbacks.metrics_callback import MetricsCallback

seed = 90
torch.manual_seed(seed)


# @dataclass
# class ModelConfig:
#     """Model configuration"""

#     input_dim: int
#     hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
#     output_dim: int = 1
#     dropout_rate: float = 0.1
#     activation: str = "ReLU"


# @dataclass
# class TrainingConfig:
#     """Training configuration"""

#     # Training params
#     batch_size: int = 32
#     num_epochs: int = 10
#     learning_rate: float = 1e-3
#     weight_decay: float = 0.01
#     warmup_steps: int = 100
#     max_grad_norm: float = 1.0

#     # Early stopping
#     patience: int = 5
#     min_delta: float = 1e-4

#     # System
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     num_workers: int = 4
#     pin_memory: bool = True
#     mixed_precision: bool = True

#     # Logging
#     log_dir: str = "logs"
#     experiment_name: str = "default_run"
#     log_interval: int = 100

#     def save(self, path: Union[str, Path]):
#         with open(path, "w") as f:
#             json.dump(self.__dict__, f, indent=2)

#     @classmethod
#     def load(cls, path: Union[str, Path]):
#         with open(path, "r") as f:
#             config_dict = json.load(f)
#         return cls(**config_dict)


@dataclass
class Config:
    n_epoch: int = 15
    max_grad_norm: int = 10000
    batch_size: int = 50
    in_features: int = 10
    hidden_dim_mult: List = field(default_factory=lambda: [2, 2])
    n_class: int = 1
    lr: float = 5e-2
    weight_decay: float = 0  # 0.01  # 0.01  # default from pytorch
    print_freq: int = 10
    l1 = 0
    l2 = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path("logs")
    within_epoch_logs_path = "./logs/within_epoch_logs.csv"
    run_description = "Single batch. Test without bells and whistle."
    experiment_name = "Single batch. Test without bells and whistle."
    save = True

    def __post_init__(self):
        """This method is automatically called after the dataclass is initialized."""
        self.setup()

    def setup(self):
        """Create the 'logs' directory if it doesn't exist."""
        if not os.path.exists(self.log_dir) and self.save:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"Created log directory: {self.log_dir}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class Model(nn.Module):
    def __init__(self, CFG: Config):
        super().__init__()
        # self.linear = nn.Linear(CFG.in_features, CFG.n_class)
        dimensions = (
            [CFG.in_features]
            + [CFG.in_features * mult for mult in CFG.hidden_dim_mult]
            + [CFG.n_class]
        )
        self.layers = nn.ModuleList()
        for in_ft, out_ft in zip(dimensions[:-1], dimensions[1:]):
            self.layers.append(nn.Linear(in_ft, out_ft))
        self.activation = nn.Sigmoid()
        self.regularization_weights = []

        # Add all model parameters
        if CFG.l1 > 0 or CFG.l2 > 0:
            self.add_regularization_weights(
                list(self.named_parameters()), l1=CFG.l1, l2=CFG.l2
            )

    def forward(self, X, y=None):
        out = X
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i + 1 != len(self.layers):
                out = nn.ReLU()(out)
        out = self.activation(out)
        loss = None
        if y is not None:
            loss = nn.functional.binary_cross_entropy(
                input=out.view(-1), target=y.float()
            )
            if len(self.regularization_weights) > 0:
                reg_loss = self.get_regularization_loss().squeeze()
                loss += reg_loss
        return out, loss

    def add_regularization_weights(self, weight_list, l1=0, l2=0):
        self.regularization_weights = [(weight_list, l1, l2)]

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,))
        for weight_list, l1, l2 in self.regularization_weights:
            for w in weight_list:
                if isinstance(w, tuple):  # if named parameter
                    w = w[1]

                if l1 > 0:
                    total_reg_loss += torch.sum(torch.abs(w) * l1)

                if l2 > 0:
                    total_reg_loss += torch.sum(l2 * torch.square(w))
        return total_reg_loss


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def calculate_auc(y_pred, y_true):
    return roc_auc_score(y_true=y_true, y_score=y_pred)


def asMinute(seconds):
    m = int(seconds // 60)
    r = int(seconds - m * 60)
    return f"{m:02d}min{r:02d}s"


def timeStat(start, percent):
    now = time.time()
    elapsed = now - start
    elapsed_per_percent = elapsed / percent

    # same as: remainder = elapsed_per_percent * (1 - percent)
    remainder = (
        elapsed_per_percent - elapsed
    )  # this beocomes (elapsed_per_step * (total_step - current_step))
    return asMinute(elapsed), asMinute(remainder)


def calculate_metric_score(y_pred, y_true):
    return calculate_auc(y_pred=y_pred, y_true=y_true).item()


# @dataclass
# class TrainingMetrics:
#     enabled: bool = None
#     store_history: bool = False
#     enabled_metrics: Union[List[str], Set[str]] = field(default_factory=list)

#     losses: AverageMeter = field(
#         default_factory=lambda: AverageMeter(store_history=True, store_avg_history=True)
#     )
#     batch_time: Optional[AverageMeter] = None
#     data_time: Optional[AverageMeter] = None
#     sent_count: Optional[AverageMeter] = None
#     scores: Optional[AverageMeter] = None
#     grad_values: Optional[AverageMeter] = None
#     lrs: Optional[AverageMeter] = None

#     def __post_init__(self):
#         """Initialize only the enabled metrics"""
#         if len(self.enabled_metrics) > 0:
#             self.enabled = True

#         for metric_name in self.enabled_metrics:
#             if hasattr(self, metric_name):
#                 setattr(
#                     self,
#                     metric_name,
#                     AverageMeter(
#                         store_history=self.store_history,
#                         store_avg_history=(metric_name == "losses"),
#                     ),
#                 )

#             else:
#                 logger.warning(
#                     "Warning, metric: {metric_name} not supported and ignored."
#                 )


#     def init_before_epoch(self):
#         self.start = self.end = time.time()
#         self.global_step = 0

#     def update_end_time(self):
#         if self.enabled:
#             self.end = time.time()
#             self.global_step += 1

#     def update(self, name: str, value: float, n: int= 1):
#         """Update metric if enabled, otherwise no-op"""
#         if self.enabled and hasattr(self, name) and getattr(self, name) is not None:
#             getattr(self, name).update(value, n)

#     def log_step(
#         self,
#         step,
#         epoch,
#         losses,
#         lrs,
#         global_step,
#         batch_time,
#         data_time,
#         sent_count,
#         scores,
#         total_steps,
#         start,
#     ):
#         if self.enabled_metrics:
#             logger.info(
#                 f"Epoch[{epoch}] "
#                 f"steps:[{step +1}/{len(train_dataloader)})] "
#                 f"total_elapsed_time: {total_elapsed}, "
#                 f"remaining_time: {remaining} "
#                 f"data_time: {metrics.data_time.val:.4f} "  # not average to monitor per batch data
#                 f"elapsed_batch_time:{metrics.batch_time.val:.4f} "
#                 f"sent_count_s: {(metrics.sent_count.avg / metrics.batch_time.avg):.3f} "  # average number of samples per second
#                 f"lr: {optimizer.param_groups[0]['lr']:.5f} "
#                 f"loss: {metrics.losses.val:.3f} "
#                 f"avg loss: {metrics.losses.avg:.3f} "
#                 f"score: {metrics.scores.val:.3f} ({metrics.scores.avg})"
#                 f"grad: {metrics.grad_values.val:.3f} "
#                 f"gradnorm: {grad_norm:.3f} "
#             )


def train(
    train_dataloader,
    model,
    optimizer,
    epoch,
    metrics_callback: TrainerCallback,
    callbacks: List[TrainerCallback] = None,
):
    model.train()

    end = time.time()
    global_step = 0
    total_steps = len(train_dataloader)
    
    for step, (bx, by) in enumerate(train_dataloader):
        # Measure data loading time
        data_time = time.time() - end

        # Send to device
        bx, by = bx.to(CFG.device), by.to(CFG.device)

        # Model out
        out, loss = model(bx, by)

        # without gradient accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # Grad checks
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )
        # keep track of current scores
        with torch.no_grad():
            model.eval()
            score = calculate_metric_score(
                y_pred=out.view(-1).detach().numpy(), y_true=by.detach().numpy()
            )
            model.train()

        batch_metrics = {
            "epoch": epoch,  # current epoch
            "total_steps": total_steps,  # total number of steps
            "data_time": data_time,  # track time taken to load a single batch of data
            "batch_time": time.time() - end,  # track time taken for a single batch
            "sent_count": {
                "value": out.shape[0],
                "n": 1,
            },  # track number of samples sent
            "losses": {"value": loss.item(), "n": out.shape[0]},  # track loss
            "scores": {"value": score, "n": out.shape[0]},  # track score
            "grad_values": grad_norm,  # track gradients
            "lrs": optimizer.param_groups[0]["lr"],  # track learning rate
        }
        metrics_callback.on_batch_end(batch=step, logs=batch_metrics)

        end = time.time()
        global_step += 1

    metrics_callback.on_training_end()

    return {
        "losses": 1,
        "lrs": 1,
        "global_step": 1,
        "batch_time": 1,
        "data_time": 1,
        "sent_count": 1,
        "scores": 1,
    }


def fast_validate(model, val_dataloader):
    model.eval()

    predictions = []
    labels = []

    for step, (bx, by) in enumerate(val_dataloader):
        # send to device for inference
        bx, by = bx.to(CFG.device), by.to(CFG.device)

        with torch.no_grad():
            out, loss = model(bx, by)

        # Do not store predictions in gpu memory to preserve memory
        predictions.append(out.detach().cpu())
        labels.append(by.detach().cpu())

    predictions = torch.cat(predictions).view(-1).numpy()
    labels = torch.cat(labels).numpy()
    final_score = calculate_metric_score(y_pred=predictions, y_true=labels)
    return final_score


def validate(model, val_dataloader, store_history=True):
    batch_time = AverageMeter(store_history=store_history)
    data_time = AverageMeter(store_history=store_history)
    losses = AverageMeter(store_history=store_history)
    sent_count = AverageMeter(store_history=store_history)
    scores = AverageMeter(store_history=store_history)

    model.eval()
    start = end = time.time()

    predictions = []
    labels = []
    global_step = 0

    for step, (bx, by) in enumerate(val_dataloader):
        # Track time taken to load a single batch of data
        data_time.update(time.time() - end, n=1)

        # send to device
        bx, by = bx.to(CFG.device), by.to(CFG.device)

        with torch.no_grad():
            out, loss = model(bx, by)
            # log accuracy
            score = calculate_metric_score(
                y_pred=out.view(-1).detach().numpy(), y_true=by.detach().numpy()
            )

        # Do not store predictions in gpu memory to preserve memory
        predictions.append(out.detach().cpu())
        labels.append(by.detach().cpu())

        # keep track of current loss
        losses.update(loss.data.item(), n=bx.shape[0])
        # track the current running score
        scores.update(score, bx.shape[0])
        # keep track of samples sent
        sent_count.update(bx.shape[0], n=1)
        # keep track of time taken to process a single batch
        batch_time.update(time.time() - end, n=1)
        end = time.time()
        global_step += 1

        if (
            (step == 0)
            or ((step + 1) % CFG.print_freq == 0)
            or (step + 1 == len(val_dataloader))
        ):
            total_elapsed, remaining = timeStat(
                start=start, percent=(step + 1) / len(train_dataloader)
            )

            logger.info(
                f"Step: [{step+1}/{len(val_dataloader)}] "
                f"total_elapsed_time: {total_elapsed} "
                f"remaining: {remaining} "
                f"data_time: {data_time.val:.4f} "
                f"elapsed_batch_time: {batch_time.val:.4f} "
                f"sent_count_s: {(sent_count.avg / batch_time.avg):.4f} "
                f"loss: {losses.val:.4f} "
                f"avg_loss: {losses.avg:.4f} "
                f"score: {scores.val:.4f} ({scores.avg:.4f}) "
            )

    predictions = torch.cat(predictions).view(-1).numpy()
    labels = torch.cat(labels).numpy()
    final_score = calculate_auc(y_pred=predictions, y_true=labels)
    logger.info(f"Final score: {final_score:.4f}")

    return {
        "losses": losses,
        "lrs": "NIL",
        "global_step": "NIL",
        "batch_time": batch_time,
        "data_time": data_time,
        "sent_count": sent_count,
        "scores": scores,
        "final_score": final_score,
    }


class TrainingLogger:
    def __init__(self, file_path, run_description):
        self.file_path = file_path
        self.run_description = run_description
        self.within_epoch_logs_dict = defaultdict(list)
        self.across_epochs_logs_dict = defaultdict(list)

    def del_file(self, path):
        if os.path.exists(path):
            os.remove(path)

    def reset(self):
        self.del_file(self.file_path)
        logger.info(f"Removed log file from : {self.file_path}")

    def update_epoch_logs(
        self, epoch_num, cur_lr, train_loss, train_score, valid_loss, valid_score
    ):
        self.across_epochs_logs_dict["epoch"].append(epoch_num)
        self.across_epochs_logs_dict["cur_lr"].append(cur_lr)
        self.across_epochs_logs_dict["train_loss"].append(train_loss)
        self.across_epochs_logs_dict["train_score"].append(train_score)
        self.across_epochs_logs_dict["valid_loss"].append(valid_loss)
        self.across_epochs_logs_dict["valid_score"].append(valid_score)

    def update_within_epoch_logs(
        self,
        epoch_num,
        losses,
        lrs,
        global_step,
        batch_time,
        data_time,
        sent_count,
        scores,
        eoe_val_loss,
        eoe_val_score,
    ):
        self.within_epoch_logs_dict["epoch"].extend(
            [epoch_num] * len(sent_count.history)
        )

        self.within_epoch_logs_dict["global_step"].extend(
            [global_step] * len(sent_count.history)
        )
        self.within_epoch_logs_dict["batch_time"].extend(batch_time.history)
        self.within_epoch_logs_dict["data_time"].extend(data_time.history)
        self.within_epoch_logs_dict["sent_count"].extend(sent_count.history)

        # training scores and loss
        self.within_epoch_logs_dict["lr"].extend(lrs.history)
        self.within_epoch_logs_dict["score"].extend(scores.history)
        self.within_epoch_logs_dict["loss_avg"].extend(losses.avg_history)
        self.within_epoch_logs_dict["loss"].extend(losses.history)

        self.within_epoch_logs_dict["eoe_val_score"].extend(
            [eoe_val_score] * len(sent_count.history)
        )
        self.within_epoch_logs_dict["eoe_val_loss"].extend(
            [eoe_val_loss] * len(sent_count.history)
        )

    def save(self):
        within_epoch_logs_df = pd.DataFrame.from_dict(self.within_epoch_logs_dict)

        # Add description to run
        within_epoch_logs_df["run_description"] = self.run_description
        within_epoch_logs_df["updated_date"] = datetime.now()

        if not os.path.exists(self.file_path):
            logger.info(
                f"path: {self.file_path} does not exist, creating new log csv of shape {within_epoch_logs_df.shape}"
            )
            within_epoch_logs_df.to_csv(self.file_path, index=False)
        else:
            # shortcut to do append: df.to_csv(csv_path, mode='a', header=False, index=False)
            logger.info(f"Appending logs to existing csv path: {self.file_path}.")

            within_epoch_logs = pd.read_csv(self.file_path)
            # conflicts in file
            if self.run_description in within_epoch_logs["run_description"].unique():
                within_epoch_logs_df["run_description"] = self.run_description + "*"

            within_epoch_logs_df = pd.concat(
                [within_epoch_logs, within_epoch_logs_df], ignore_index=True
            )
            within_epoch_logs_df.to_csv(self.file_path, index=False)
            logger.info(
                f"Log file increased from shape: {within_epoch_logs.shape} to shape: {within_epoch_logs_df.shape}."
            )
        return within_epoch_logs_df

    def get_across_epoch_logs(self, file_path):
        within_epoch_logs = pd.read_csv(file_path)

        # select for each epoch the last index (last step)
        epoch_logs = within_epoch_logs.iloc[
            within_epoch_logs.reset_index().groupby("epoch")["index"].max().tolist(), :
        ]
        return epoch_logs

    def save_across_epochs_log(self, file_path, overwrite=True):
        self.del_file(file_path)
        across_epoch_logs_df = pd.DataFrame.from_dict(self.across_epochs_logs_dict)
        across_epoch_logs_df

        return across_epoch_logs_df


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def reset_log_step():
    if os.path.exists(Path(CFG.log_dir) / "step_metrics.csv"):
        os.remove(Path(CFG.log_dir) / "step_metrics.csv")


# def _log_step(
#     step: int,
#     epoch: int,
#     losses: AverageMeter,
#     lrs: AverageMeter,
#     global_step: int,
#     batch_time: AverageMeter,
#     data_time: AverageMeter,
#     sent_count: AverageMeter,
#     scores: AverageMeter,
#     total_steps: int,
#     start: float,
# ):
#     """
#     Log metrics at step level
#     """
#     # # Prepare step metrics
#     # step_metrics = {
#     #     "step": step,
#     #     "epoch": epoch,
#     #     "global_step": self.global_step,
#     #     "lr": self.optimizer.param_groups[0]["lr"],
#     #     "batch_loss": batch_metrics["loss"],
#     #     "batch_score": batch_metrics["score"],
#     #     "batch_time": batch_metrics["batch_time"],
#     #     "samples_per_sec": batch_metrics["samples_per_sec"],
#     # }

#     # # Console logging for steps
#     # self.console_logger.info(
#     #     f"Epoch: {epoch}/{self.config.n_epoch} "
#     #     f"Step: [{step}/{batch_metrics['total_steps']}] "
#     #     f"Loss: {step_metrics['batch_loss']:.4f} "
#     #     f"Score: {step_metrics['batch_score']:.4f} "
#     #     f"LR: {step_metrics['lr']:.6f} "
#     #     f"Speed: {step_metrics['samples_per_sec']:.1f} samples/sec"
#     # )

#     total_elapsed, remaining = timeStat(
#         start=start, percent=(step + 1) / len(train_dataloader)
#     )

#     # May be slow since it is IO bounded
#     step_metrics = {}

#     step_metrics["epoch"] = epoch

#     step_metrics["global_step"] = global_step
#     step_metrics["batch_time"] = batch_time.history
#     step_metrics["data_time"] = data_time.history
#     step_metrics["sent_count"] = sent_count.history
#     # training scores and loss
#     step_metrics["lr"] = lrs.history
#     step_metrics["score"] = scores.history
#     step_metrics["loss_avg"] = losses.avg_history
#     step_metrics["loss"] = losses.history

#     # step_metrics["eoe_val_score"] = eoe_val_score
#     # step_metrics["eoe_val_loss"] = eoe_val_loss

#     logger.info(
#         f"Epoch: [{epoch}/{CFG.n_epoch}] "
#         f"Step: [{step + 1}/{total_steps}] "
#         f"total_elapsed_time: {total_elapsed} "
#         f"remaining: {remaining} "
#         f"data_time: {data_time.val:.4f} "
#         f"elapsed_batch_time: {batch_time.val:.4f} "
#         f"sent_count_s: {(sent_count.avg / batch_time.avg):.4f} "
#         f"batch_loss: {losses.val:.4f} "
#         f"batch_avg_loss: {losses.avg:.4f} "
#         f"score: {scores.val:.4f} ({scores.avg:.4f}) "
#     )

#     # # TensorBoard logging for steps
#     # for name, value in step_metrics.items():
#     #     if isinstance(value, (int, float)):
#     #         self.tb_writer.add_scalar(f'step/{name}', value, self.global_step)

#     # CSV logging for steps
#     step_log_path = CFG.log_dir / "step_metrics.csv"
#     step_df = pd.DataFrame([step_metrics])

#     if step_log_path.exists():
#         step_df.to_csv(step_log_path, mode="a", header=False, index=False)
#     else:
#         step_df.to_csv(step_log_path, index=False)


if __name__ == "__main__":
    CFG = Config()
    X = torch.randn(
        CFG.batch_size * 1, CFG.in_features
    )  # change multipler to simulate simple batch testing

    # y = torch.randint(0, 2, size=(CFG.batch_size * 100,))
    y = (X.sum(axis=-1) > X.std() * 0.1).float().view(-1)
    # print(X.requires_grad, y.requires_grad)

    train_dataset = Dataset(X=X, y=y)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=CFG.batch_size, shuffle=False
    )

    val_dataset = Dataset(X=X, y=y)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=CFG.batch_size, shuffle=False
    )
    model = Model(CFG).to(CFG.device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # do not use weight decay for biases and layernorms. Weight decay is L2 norm
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    train_logger = TrainingLogger(
        CFG.within_epoch_logs_path, run_description=CFG.run_description
    )

    train_logger.reset()
    reset_log_step()

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    metrics_callback = MetricsCallback(
        metrics_to_track=[
            "losses",
            "batch_time",
            "data_time",
            "sent_count",
            "scores",
            "grad_values",
            "lrs",
        ],
        log_dir=CFG.log_dir,
        experiment_name=CFG.experiment_name,
        store_history=True,
    )

    # initialize metric callback
    metrics_callback.on_training_start()
    
    for i in range(CFG.n_epoch):
        # metrics callback
        metrics_callback.on_epoch_start()
        # you need to initialize scheduler again if using learning rate scheduler
        # num_training_steps is for when your dataset is infinite and you want to stop

        # Train for single epoch and obtain logs for print_freq steps
        train_logs = train(
            train_dataloader,
            model,
            optimizer,
            epoch=i,
            metrics_callback=metrics_callback,
        )

        # Validate for single epoch
        # val_logs = validate(model, val_dataloader)
        # print(f'val loss: {val_logs['scores'].avg}')

        # # store training steps log
        # train_logger.update_within_epoch_logs(
        #     epoch_num=i + 1,
        #     **train_logs,
        #     eoe_val_loss=val_logs["losses"].avg,
        #     eoe_val_score=val_logs["final_score"],
        # )

        # if early_stopping(val_logs["losses"].avg):
        #     logger.info(f"Early stopping triggered at epoch: {i+1}")
        #     break

        # store training epoch logs
        metrics_callback.on_epoch_end()
        print("==========")
        # initialize metric callback
    metrics_callback.on_training_end()

    # key_len = list((k, len(l)) for k, l in train_logger.within_epoch_logs_dict.items())
    # print(key_len)

    # # plot training losses
    # total_loss_steps = train_logger.within_epoch_logs_dict["loss_avg"]
    # plt.plot(range(len(total_loss_steps)), total_loss_steps)
    # plt.show()

    # train_logger.save()
    # d = train_logger.get_across_epoch_logs(CFG.within_epoch_logs_path)


# class Trainer:
#     def __init__(self, config: TrainingConfig):
#         self.config = config
#         self.model = None
#         self.optimizer = None
#         self.scheduler = None
#         self.scaler = torch.cuda.amp.GradScaler()
#         self.early_stopping = EarlyStopping(
#             patience=config.early_stopping_patience,
#             mode="max",  # Since we're monitoring AUC score
#         )
#         self.best_model_state = None
#         self.setup_logging()

#     def setup_logging(self):
#         """Initialize logging mechanisms"""
#         # Create logging directory with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.log_dir = (
#             Path(self.config.log_dir) / self.config.experiment_name / timestamp
#         )
#         self.log_dir.mkdir(parents=True, exist_ok=True)

#         # Initialize loggers
#         self.console_logger = logging.getLogger(__name__)
#         self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

#     def train(self, train_loader, val_loader):
#         """Main training loop with early stopping"""
#         self.model = Model(self.config).to(self.config.device)
#         self.optimizer = self._create_optimizer()
#         self.scheduler = self._create_scheduler(len(train_loader))

#         for epoch in range(self.config.n_epoch):
#             # Training phase
#             train_metrics = self._train_epoch(train_loader, epoch)

#             # Validation phase
#             val_metrics = self._validate(val_loader, epoch)

#             # Log metrics
#             self._log_epoch(epoch, train_metrics, val_metrics)

#             # Early stopping check
#             if self.early_stopping(epoch, val_metrics["score"]):
#                 self.console_logger.info(
#                     f"Early stopping triggered. Best score: {self.early_stopping.best_score:.4f} "
#                     f"at epoch {self.early_stopping.best_epoch}"
#                 )
#                 # Restore best model
#                 self.model.load_state_dict(self.best_model_state)
#                 break

#             # Save best model
#             if val_metrics["score"] == self.early_stopping.best_score:
#                 self.best_model_state = copy.deepcopy(self.model.state_dict())
#                 self.save_checkpoint(
#                     self.log_dir / "best_model.pt", epoch, val_metrics["score"]
#                 )

#         return {
#             "best_score": self.early_stopping.best_score,
#             "best_epoch": self.early_stopping.best_epoch,
#         }

#     def save_checkpoint(self, path: Path, epoch: int, score: float):
#         """Save model checkpoint"""
#         checkpoint = {
#             "epoch": epoch,
#             "model_state_dict": self.model.state_dict(),
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "scheduler_state_dict": self.scheduler.state_dict()
#             if self.scheduler
#             else None,
#             "score": score,
#             "config": self.config.__dict__,
#         }
#         torch.save(checkpoint, path)
#         self.console_logger.info(f"Saved checkpoint to {path}")

#     def load_checkpoint(self, path: Path):
#         """Load model checkpoint"""
#         checkpoint = torch.load(path)
#         self.model.load_state_dict(checkpoint["model_state_dict"])
#         self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         if self.scheduler and checkpoint["scheduler_state_dict"]:
#             self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
#         return checkpoint["epoch"], checkpoint["score"]


# def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
#     """
#     Log metrics for the epoch to console, TensorBoard, and CSV.

#     Args:
#         epoch: Current epoch number
#         train_metrics: Dictionary containing training metrics
#         val_metrics: Dictionary containing validation metrics
#     """
#     # Prepare metrics dictionary
#     metrics = {
#         "epoch": epoch,
#         "lr": self.optimizer.param_groups[0]["lr"],
#         "train_loss": train_metrics["loss"],
#         "train_score": train_metrics["score"],
#         "val_loss": val_metrics["loss"],
#         "val_score": val_metrics["score"],
#         "batch_time": train_metrics["batch_time"].avg,
#         "data_time": train_metrics["data_time"].avg,
#         "samples_per_sec": train_metrics["sent_count"].sum
#         / train_metrics["batch_time"].sum,
#     }

#     # 1. Console Logging
#     self.console_logger.info(
#         f"\nEpoch [{epoch}/{self.config.n_epoch}] "
#         f"LR: {metrics['lr']:.6f}\n"
#         f"Train Loss: {metrics['train_loss']:.4f} | "
#         f"Train Score: {metrics['train_score']:.4f}\n"
#         f"Val Loss: {metrics['val_loss']:.4f} | "
#         f"Val Score: {metrics['val_score']:.4f}\n"
#         f"Best Val Score: {self.early_stopping.best_score:.4f} "
#         f"(epoch {self.early_stopping.best_epoch})\n"
#         f"Batch time: {metrics['batch_time']:.3f}s | "
#         f"Data time: {metrics['data_time']:.3f}s | "
#         f"Samples/sec: {metrics['samples_per_sec']:.1f}\n"
#         + (
#             "★ New best score! "
#             if metrics["val_score"] == self.early_stopping.best_score
#             else ""
#         )
#     )

#     # 2. TensorBoard Logging
#     # Scalars
#     for name, value in metrics.items():
#         if isinstance(value, (int, float)):
#             self.tb_writer.add_scalar(f"epoch/{name}", data=value, step=epoch)

#     # Learning rate
#     self.tb_writer.add_scalar("epoch/learning_rate", data=metrics["lr"], step=epoch)

#     # Gradients (optional)
#     if hasattr(self, "log_gradients") and self.log_gradients:
#         for name, param in self.model.named_parameters():
#             if param.grad is not None:
#                 self.tb_writer.add_histogram(
#                     f"gradients/{name}", data=param.grad, step=epoch
#                 )

#     # 3. CSV Logging
#     csv_path = self.log_dir / "metrics.csv"
#     csv_metrics = pd.DataFrame([metrics])

#     if csv_path.exists():
#         # Append to existing CSV
#         csv_metrics.to_csv(csv_path, mode="a", header=False, index=False)
#     else:
#         # Create new CSV
#         csv_metrics.to_csv(csv_path, index=False)

#     # 4. Update Training Logger (if using)
#     if hasattr(self, "training_logger"):
#         # self.training_logger.update_epoch_logs(
#         #     epoch_num=epoch,
#         #     cur_lr=metrics["lr"],
#         #     train_loss=metrics["train_loss"],
#         #     train_score=metrics["train_score"],
#         #     valid_loss=metrics["val_loss"],
#         #     valid_score=metrics["val_score"],
#         # )

#         self.training_logger.update_within_epoch_logs(
#             epoch_num=epoch,
#             losses=train_metrics["losses"],
#             lrs=train_metrics["lrs"],
#             global_step=train_metrics["global_step"],
#             batch_time=train_metrics["batch_time"],
#             data_time=train_metrics["data_time"],
#             sent_count=train_metrics["sent_count"],
#             scores=train_metrics["scores"],
#             eoe_val_loss=metrics["val_loss"],
#             eoe_val_score=metrics["val_score"],
#         )

#         # Save training logger periodically
#         if epoch % self.config.save_freq == 0 or epoch == self.config.n_epoch - 1:
#             self.training_logger.save()

#     return metrics
