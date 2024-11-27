import copy
import json
import os
import sys
import time
from collections import defaultdict
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

from tabular_reusable_assets.config.training_arguments import TrainingArguments
from tabular_reusable_assets.utils.logger import default_logger as logger

from .callbacks.early_stopping_callback import EarlyStoppingCallback
from .callbacks.metrics_callback import MetricsCallback
from .callbacks.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

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


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"


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
    save_epoch_freq: int = 1

    def __post_init__(self):
        """This method is automatically called after the dataclass is initialized."""
        self.setup()

    def setup(self):
        """Create the 'logs' directory if it doesn't exist."""
        if not os.path.exists(self.log_dir) and self.save:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"Created log directory: {self.log_dir}")

    def to_dict(self):
        dict = {}
        for key, value in self.__dict__.items():
            dict[key] = value
        return dict

    @classmethod
    def from_dict(cls):
        return cls(**dict)


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


def train(
    train_dataloader,
    model,
    optimizer,
    epoch,
    metrics_callback: TrainerCallback,
    callbacks: List[TrainerCallback] = None,
    callback_handler: CallbackHandler = None,
    args: TrainingArguments = None,
    state: TrainerState = None,
    control: TrainerControl = None,
):
    model.train()

    end = time.time()
    global_step = 0
    steps_in_epoch = len(train_dataloader)

    for step, (bx, by) in enumerate(train_dataloader):
        # Measure data loading time
        data_time = time.time() - end

        # Send to device
        bx, by = bx.to(CFG.device), by.to(CFG.device)

        # Model out
        out, loss = model(bx, by)

        callback_handler.on_step_begin(args, state, control)
        control = callback_handler.on_pre_optimizer_step(args, state, control)

        # without gradient accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        control = callback_handler.on_optimizer_step(args, state, control)

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

        n_batch_sample = out.shape[0]
        batch_metrics = {
            "epoch": epoch,  # current epoch
            "total_steps": steps_in_epoch,  # total number of steps
            "data_time": data_time,  # track time taken to load a single batch of data
            "batch_time": time.time() - end,  # track time taken for a single batch
            "sent_count": {
                "value": n_batch_sample,
                "n": 1,
            },  # track number of samples sent
            "losses": loss.item(),  # {"value": loss.item(), "n": n_batch_sample},  # track loss
            "scores": {"value": score, "n": n_batch_sample},  # track score
            "grad_values": grad_norm.item(),  # track gradients
            "lrs": optimizer.param_groups[0]["lr"],  # track learning rate
        }

        # print(batch_metrics, type(batch_metrics['sent_count']['value']), type(batch_metrics['losses']['value']), type(batch_metrics['scores']['value']))
        
      
        
        end = time.time()

        state.global_step += 1
        state.epoch = epoch + (step + 1) / steps_in_epoch

        control = callback_handler.on_step_end(args, state, control)
        metrics_callback.on_step_end(batch=step, logs=batch_metrics)


        if control.should_save:
            logger.info("Saving model")

        if control.should_evaluate:
            logger.info("Evaluating model")

        if control.should_log:
            logger.info("Logging model")

    metrics_callback.on_train_end()
    state.log_history.append(batch_metrics)
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


@dataclass
class ModelConfig:
    hidden_layers = [1, 2, 3]

    def to_dict(self):
        return {"hidden_layers": self.hidden_layers}

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


def _save(
    model,
    optimizer,
    lr_scheduler,
    config,
    args: TrainingArguments,
    state: TrainerState,
    control: TrainerControl,
):
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    state.stateful_callbacks["TrainerControl"] = control.state()
    # state.stateful_callbacks['TrainingArguments'] # this is something that users will change, so dont save this.

    if state:
        state.save_to_json(Path(args.output_dir) / TRAINER_STATE_NAME)

    if config:
        torch.save(config.to_dict(), Path(args.output_dir) / CONFIG_NAME)

    # Save model
    if model:
        torch.save(model.state_dict(), Path(args.output_dir) / WEIGHTS_NAME)

    # Save Optimizer
    if optimizer:
        torch.save(optimizer.state_dict(), Path(args.output_dir) / OPTIMIZER_NAME)

    # Save Scheduler
    if lr_scheduler:
        torch.save(lr_scheduler.state_dict(), Path(args.output_dir) / SCHEDULER_NAME)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, Path(args.output_dir) / TRAINING_ARGS_NAME)
    return


# def _load(model, optimizer, lr_scheduler, config, args, state, control):
#     model_path = Path(args.output_dir) / WEIGHTS_NAME
#     optimizer_path = Path(args.output_dir) / OPTIMIZER_NAME
#     lr_scheduler_path = Path(args.output_dir) / SCHEDULER_NAME
#     args_path = Path(args.output_dir) / TRAINING_ARGS_NAME
#     trainer_state_path = Path(args.output_dir) / TRAINER_STATE_NAME

#     model.set_state


def _maybe_log_save_evaluate(
    args, state, control, loss, gradnorm, model, epoch, optimizer, lr_scheduler
):
    if control.should_log:
        print("should log here")

    if control.should_save:
        print("should_save_here")
    return


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

    # train_logger.reset()
    # reset_log_step()

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        metric_for_best_model="losses",
        greater_is_better=False,
        logging_steps=10,
        save_steps=5,
        output_dir="./data/output_dir",
    )
    state = TrainerState()
    state.max_steps = len(train_dataloader) * CFG.n_epoch
    control = TrainerControl()

    early_stopping_callback = EarlyStoppingCallback()
    callbacks = [DefaultFlowCallback(), ProgressCallback(), early_stopping_callback]

    callback_handler = CallbackHandler(
        callbacks, model, processing_class=None, optimizer=optimizer, lr_scheduler=None
    )
    # callback_handler.add_callback()
    callback_handler.pop_callback(ProgressCallback)

    control = callback_handler.on_train_begin(args, state, control)

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
        state=state,
    )

    # initialize metric callback
    metrics_callback.on_train_begin()

    early_stopping_callback.on_train_begin(args, state=state, control=control)

    for i in range(CFG.n_epoch):
        # metrics callback
        metrics_callback.on_epoch_begin()
        control = callback_handler.on_epoch_begin(args, state, control)

        # you need to initialize scheduler again if using learning rate scheduler
        # num_training_steps is for when your dataset is infinite and you want to stop

        # Train for single epoch and obtain logs for print_freq steps
        train_logs = train(
            train_dataloader,
            model,
            optimizer,
            epoch=i,
            metrics_callback=metrics_callback,
            callback_handler=callback_handler,
            state=state,
            args=args,
            control=control,
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

        # Save current model at every epoch
        # model_path = CFG.log_dir / f"checkpoints/model_epoch_{i}.pt"
        # torch.save(model.state_dict(), model_path)

        state.epoch += 1
        control = callback_handler.on_epoch_end(args, state, control)

        # store training epoch logs
        metrics_callback.on_epoch_end(
            current_epoch_loss=metrics_callback.metrics.losses.avg,
            current_epoch_score=metrics_callback.metrics.scores.val,
            model_path=None,
        )

        early_stopping_callback.on_evaluate(
            args,
            state=metrics_callback.state,
            control=control,
            metrics=metrics_callback.metrics,
        )

        control = callback_handler.on_evaluate(
            args, state, control, metrics=metrics_callback.metrics
        )

        if control.should_training_stop:
            break

        if control.should_save:
            _save(
                model,
                optimizer,
                lr_scheduler=None,
                config=CFG,
                args=args,
                state=state,
                control=control,
            )

        # time.sleep(2)
        # logger.info("==========")

    control = callback_handler.on_train_end(args, state, control)
    print(early_stopping_callback.state())
    # initialize metric callback
    metrics_callback.on_train_end()
    # logger.info(f"Trainer State: {state.to_dict()}")

    # key_len = list((k, len(l)) for k, l in train_logger.within_epoch_logs_dict.items())
    # print(key_len)

    # # plot training losses
    # total_loss_steps = train_logger.within_epoch_logs_dict["loss_avg"]
    # plt.plot(range(len(total_loss_steps)), total_loss_steps)
    # plt.show()

    # train_logger.save()
    # d = train_logger.get_across_epoch_logs(CFG.within_epoch_logs_path)
