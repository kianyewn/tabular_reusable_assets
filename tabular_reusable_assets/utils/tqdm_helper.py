import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm  # Automatically choose between notebook and console


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm to avoid breaking progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class ProductionProgressBar:
    """Production-ready progress bar wrapper with logging integration."""

    def __init__(
        self,
        total: Optional[int] = None,
        desc: str = "",
        log_file: str = "progress.log",
        log_level: int = logging.INFO,
        disable: bool = False,
    ):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers = []

        # Add handlers for both file and tqdm-compatible console output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self.logger.addHandler(file_handler)
        self.logger.addHandler(TqdmLoggingHandler())

        # Initialize progress bar
        self.pbar = auto_tqdm(
            total=total,
            desc=desc,
            disable=disable,
            file=sys.stdout,
            dynamic_ncols=True,  # Automatically adjust to terminal width
            mininterval=0.5,  # Minimum time between updates
            maxinterval=2.0,  # Maximum time between updates
            smoothing=0.3,  # Exponential moving average smoothing factor
        )

        # Store initial parameters
        self.total = total
        self.desc = desc

    def update(self, n: int = 1, postfix: dict = None):
        """Update progress bar with optional postfix dict for metrics."""
        self.pbar.update(n)
        if postfix:
            self.pbar.set_postfix(postfix, refresh=True)

    def log(self, message: str, level: int = logging.INFO):
        """Log a message without breaking the progress bar."""
        self.logger.log(level, message)

    def close(self):
        """Properly close the progress bar."""
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            self.log(f"Error occurred: {exc_val}", logging.ERROR)
            return False
        return True


@contextmanager
def progress_bar_context(iterable: Iterator[Any] = None, **kwargs) -> Iterator[tqdm]:
    """Context manager for progress bars with automatic cleanup."""
    pbar = None
    try:
        if iterable is not None:
            pbar = ProductionProgressBar(total=len(iterable), **kwargs)
            yield pbar
        else:
            pbar = ProductionProgressBar(**kwargs)
            yield pbar
    finally:
        if pbar is not None:
            pbar.close()


# Example usage scenarios:


def example_basic_usage():
    """Basic usage with context manager."""
    with ProductionProgressBar(total=100, desc="Processing") as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)
            pbar.log(f"Completed iteration {i}")


def example_iteration():
    """Iterating over a collection with progress tracking."""
    items = list(range(100))
    with progress_bar_context(desc="Processing items") as pbar:
        for item in items:
            time.sleep(0.01)
            pbar.update(1)
            if item % 10 == 0:
                pbar.log(f"Processed milestone: {item}")


def example_with_metrics():
    """Progress bar with metrics tracking."""
    with ProductionProgressBar(total=100, desc="Training") as pbar:
        for epoch in range(10):
            loss = 1.0 / (epoch + 1)
            accuracy = epoch / 10
            pbar.update(10, {"loss": f"{loss:.4f}", "acc": f"{accuracy:.2%}"})
            pbar.log(f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.2%}")


def example_nested_bars():
    """Handling nested progress bars."""
    with ProductionProgressBar(total=5, desc="Outer") as pbar_outer:
        for i in range(5):
            with ProductionProgressBar(total=20, desc=f"Inner-{i}") as pbar_inner:
                for j in range(20):
                    time.sleep(0.01)
                    pbar_inner.update(1)
            pbar_outer.update(1)


if __name__ == "__main__":
    # Example of production usage
    try:
        with ProductionProgressBar(total=100, desc="Main Process") as pbar:
            # Simulate some work
            for i in range(10):
                time.sleep(0.1)
                if i == 5:
                    # Log a warning
                    pbar.log("Warning: Halfway point reached", logging.WARNING)

                # Update progress with metrics
                pbar.update(10, {"iteration": i, "status": "running"})

            # Log successful completion
            pbar.log("Process completed successfully")

    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise
