import time

from tabular_reusable_assets.utils.tqdm_helper import ProductionProgressBar, progress_bar_context


# Example usage scenarios:


def test_example_basic_usage():
    """Basic usage with context manager."""
    with ProductionProgressBar(total=10, desc="Processing") as pbar:
        for i in range(10):
            time.sleep(0.01)
            pbar.update(10)
            pbar.log(f"Completed iteration {i}")


def test_example_iteration():
    """Iterating over a collection with progress tracking."""
    items = list(range(10))
    with progress_bar_context(desc="Processing items") as pbar:
        for item in items:
            time.sleep(0.01)
            pbar.update(1)
            if item % 10 == 0:
                pbar.log(f"Processed milestone: {item}")


def test_example_with_metrics():
    """Progress bar with metrics tracking."""
    with ProductionProgressBar(total=10, desc="Training") as pbar:
        for epoch in range(10):
            loss = 1.0 / (epoch + 1)
            accuracy = epoch / 10
            pbar.update(10, {"loss": f"{loss:.4f}", "acc": f"{accuracy:.2%}"})
            pbar.log(f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.2%}")


def test_example_nested_bars():
    """Handling nested progress bars."""
    with ProductionProgressBar(total=5, desc="Outer") as pbar_outer:
        for i in range(5):
            with ProductionProgressBar(total=20, desc=f"Inner-{i}") as pbar_inner:
                for j in range(20):
                    time.sleep(0.01)
                    pbar_inner.update(1)
            pbar_outer.update(1)
