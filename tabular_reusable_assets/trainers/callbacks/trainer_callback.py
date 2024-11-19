from abc import ABC

class TrainerCallback(ABC):
    """Abstract base class for callbacks"""
    def on_training_start(self): pass
    def on_training_end(self): pass
    def on_epoch_start(self): pass
    def on_epoch_end(self, epoch: int, logs: dict = None): pass
    def on_batch_start(self, batch: int, logs: dict = None): pass
    def on_batch_end(self, batch: int, logs: dict = None): pass

    
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
    should_epoch_stop: bool 
    should_log: bool 
    should_evaluate: bool 
    should_save: bool 

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

