from abc import ABC

class TrainerCallback(ABC):
    """Abstract base class for callbacks"""
    def on_training_start(self): pass
    def on_training_end(self): pass
    def on_epoch_start(self): pass
    def on_epoch_end(self, epoch: int, logs: dict = None): pass
    def on_batch_start(self, batch: int, logs: dict = None): pass
    def on_batch_end(self, batch: int, logs: dict = None): pass