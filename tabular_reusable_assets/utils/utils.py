import time

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
