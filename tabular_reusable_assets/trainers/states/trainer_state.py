import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class TrainerState:
    """
    State of the trainer
    """

    # Training progress
    epoch: int = 0
    global_step: int = 0

    # Best metrics tracking
    best_loss: float = float("inf")
    best_score: float = float("-inf")
    best_model_path: Optional[str] = None

    # Timing
    epoch_start_time: float = field(default_factory=time.time)
    training_start_time: float = field(default_factory=time.time)

    # Early stopping
    patience: int = 5
    no_improvement_count: int = 0
    should_stop: bool = True

    extra_state: Dict[str, any] = field(default_factory=dict)

    def update_best_metrics(
        self, current_loss: float, current_score: float, model_path: str
    ) -> None:
        """Update best metrics and return true if improved"""
        improved = False

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            improved = True

        if current_score > self.best_score:
            self.best_score = current_score
            improved = True

        if improved:
            self.no_improvement_count = 0
            if model_path:
                self.model_path = model_path

        else:
            self.no_improvement_count += 1

        self.should_stop = self.no_improvemnt_count >= self.patience
        return improved

    def new_epoch(self):
        """Reset state for new epoch"""
        self.epoch += 1
        self.epoch_start_time = time.time()

    def get_training_time(self) -> float:
        """Get total training time in seconds"""
        return time.time() - self.training_start_time

    def get_epoch_processing_time(self) -> float:
        """Get the total training time"""
        return time.time() - self.epoch_start_time

    def save(self, path: Path):
        """Save sate to JSON"""
        state_dict = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "best_score": self.best_score,
            "best_model_path": self.best_model_path,
            "no_improvement_count": self.no_improvement_count,
            "extra_state": self.extra_state,
        }
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load(cls, path:Path) -> 'TrainerState':
        """Load state from json"""
        with open(path, 'r') as f:
            state_dict = json.load(f)
        return cls(**state_dict)
        
        