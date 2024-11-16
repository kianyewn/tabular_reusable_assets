from dataclasses import dataclass, field
import torch
from typing import List

@dataclass
class AverageMeter:
    """Computes and stores the average and current value of metrics.
    
    Attributes:
        store_history: If True, stores all individual values
        store_avg_history: If True, stores running average history
        val: Current value
        count: Number of updates
        total: Sum of all values
        avg: Running average
        history: List of individual values (if enabled)
        avg_history: List of running averages (if enabled)
    """
    store_history: bool = False
    store_avg_history: bool = False
    val: float = field(default=0.0, init=False)
    count: int = field(default=0, init=False)
    total: float = field(default=0.0, init=False)
    avg: float = field(default=0.0, init=False)
    history: List[float] = field(default_factory=list, init=False)
    avg_history: List[float] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Resets all accumulated values to zero."""
        self.val = 0.0
        self.count = 0
        self.total = 0.0
        self.avg = 0.0
        self.history.clear()
        self.avg_history.clear()

    def update(self, val: float, n: int = 1) -> None:
        """Updates the meter with new values.
        
        Args:
            val: The value to add
            n: Number of items this value represents (default: 1)
            
        Raises:
            ValueError: If n is less than 1 or val is not a number
        """
        if not isinstance(val, (int, float, torch.Tensor)):
            raise ValueError(f"Value must be a number, got {type(val)}")

        if n < 1:
            raise ValueError(f"Update count must be positive, got {n}")
        
        self.val = float(val)
        self.count += n
        self.total += val * n
        self.avg = self.total / self.count

        if self.store_history:
            self.history.append(self.val)
            if self.store_avg_history:
                self.avg_history.append(self.avg)
