from abc import ABC
from dataclasses import dataclass

from tabular_reusable_assets.utils.logger import default_logger as logger


class TrainerCallback(ABC):
    """Abstract base class for callbacks"""

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        pass

    def on_batch_start(self, batch: int, logs: dict = None):
        pass

    def on_batch_end(self, batch: int, logs: dict = None):
        pass


class ExportableState:
    def state(self) -> dict:
        raise NotImplementedError(
            "You msut implement a `state` function to utilize this class"
        )

    @classmethod
    def set_state(cls, state: dict):
        instance = cls(**state["args"])
        for k, v in state["attributes"].items():
            setattr(instance, k, v)
        return instance

    def state_example(self) -> dict:
        return {
            "args": {"arg1": "hello_world", "arg2": 123},
            "attributes": {"hello": "world", "bonjour": "le monde"},
        }


@dataclass
class TrainerControl:
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_log: bool = False
    should_evaluate: bool = False
    should_save: bool = False

    def _new_training(self):
        """Internal method that resets the variable for new training"""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch"""
        self.should_epoch_stop = False

    def _new_step(self):
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False

    def state(self):
        return {
            "args": {
                "should_training_stop": self.should_training_stop,
                "should_epoch_stop": self.should_epoch_stop,
                "should_log": self.should_log,
                "should_evaluate": self.should_evaluate,
                "should_save": self.should_save,
            },
            "attributes": {},
        }


class CallbackHandler(TrainerCallback):
    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler):
        self.callbacks = []
        for callback in callbacks:
            self.add_callback(callback)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

    def add_callback(self, callback):
        # if callback is already an instance, use as it is
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [cb.__class__ for cb in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is \n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.rmeove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb
            # return self.callbacks.pop(self.callbacks.index(callback))

    def remove_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        # cb_class = callback if isinstance(callback, type) else callback.__class__
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
        else:
            self.callbacks.remove(cb)

    @property
    def callback_list(self):
        return [cb.__class__.__name__ for cb in self.callbacks]

    def on_init_end(self, args, state, control, **kwargs):
        return self.call_event("on_init_end", args, state, control, **kwargs)

    def on_train_begin(self, args, state, control):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_start(self, args, state, control, **kwargs):
        return self.call_event(
            '"on_train_start', args, state, control, **kwargs)

    def on_train_end(self, args, state, control):
        return self.call_event('on_train_end', args, state, control)
    
    def on_epoch_begin(self, args, state, control):
        control.should_epoch_stop = False
        return self.call_event('on_epoch_begin', args, state, control)
    
    def on_epoch_end(self, args, state, control):
        return self.call_event('on_epoch_end', args, state, control)
    
    def on_step_begin(self, args, state, control):
       control.should_log = False
       control.should_evaluate = False
       control.should_save = False
       return self.call_event('on_step_begin', args, state, control)

    def on_pre_optimizer_step(self, args, state, control):
        return self.call_event('on_pre_optimizer_step', args, state, control)

    def on_optimizer_step(self, args, state, control):
        return self.call_evnet['on_optimizer_step',args, state, control]

    def on_substep_end(self, args, state, control):
        return self.call_event("on_step_end", args, state, control)
    
    def on_step_end(self, args, state, control):
        return self.call_event('on_step_end', args, state, control)

    def on_evaluate(self, args, state, control, metrics):
        control.should_evaluate = False
        return self.call_event('on_evaluate', args, state, control, metrics=metrics)
    
    def on_predict(self, args, state, control, metrics):
        return self.call_event('on_predict', args, state, control, metrics=metrics) 

    def on_save(self, args, state, control):
        control.should_save = False
        return self.call_event('on_save', args, state, control)
    
    def on_log(self, args, state, control, logs):
        control.should_log = False
        return self.call_event('on_log', args, state, control, logs=logs)
        
    def on_prediction_step(self, args, state, control):
        return self.call_event('on_prediction_step', args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            # basically for callback `event`, the callback event function MUST accept **kwargs
            # because we are providing other objects stored in the callback handler`
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A callback can skip the return of 'control' if it doesnt change it
            if result is not None:
                control = result
        return control
