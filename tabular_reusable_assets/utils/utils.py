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


class Timer(object):
    """Timer class.

    `Original code <https://github.com/miguelgfierro/pybase/blob/2298172a13fb4a243754acbc6029a4a2dcf72c20/log_base/timer.py>`_.

    Examples:
        >>> import time
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> with Timer() as t:
        ...   time.sleep(1)
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 1...'
    """

    def __init__(self):
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = time.time()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = time.time()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )

    @property
    def interval(self):
        """Get time interval in seconds.

        Returns:
            float: Seconds.
        """
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            total_seconds = self._interval
            return total_seconds

    @property
    def elapsed_seconds(self):
        return self._interval

    @property
    def elapsed_minute_seconds(self):
        total_seconds = self._interval
        minutes = int(total_seconds // 60)
        remainder_seconds = int(total_seconds - minutes * 60)
        return f"{minutes:02d}min{remainder_seconds:02d}seconds"
