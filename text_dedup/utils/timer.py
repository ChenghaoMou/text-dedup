#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:45:46
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import time


class TimerContext:
    def __init__(self, timer: "Timer", name: str):
        self.timer = timer
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if any([exc_type, exc_val, exc_tb]):
            raise exc_val
        self.timer.elapsed_times[self.name] = time.time() - self.start_time


class Timer:
    """
    A simple timer that tracks the elapsed time of each context.

    Examples
    --------
    >>> t = Timer()
    >>> with t("test"):
    ...     time.sleep(1)
    >>> assert int(t.elapsed_times.get("test", 0)) >= 1, "The elapsed time should be 1 second."
    """

    def __init__(self):
        self.elapsed_times = {}

    def __call__(self, name: str) -> TimerContext:
        """
        Create a context with the given name.

        Parameters
        ----------
        name: str
            The name of the context.

        Returns
        -------
        TimerContext
            The context.

        Examples
        --------
        >>> t = Timer()
        >>> with t("test"):
        ...     time.sleep(1)
        >>> assert int(t.elapsed_times.get("test", 0)) == 1, "The elapsed time should be 1 second."
        >>> with t("test2"):
        ...     time.sleep(2)
        >>> assert int(t.elapsed_times.get("test2", 0)) == 2, "The elapsed time should be 2 seconds."
        """
        return TimerContext(self, name)
