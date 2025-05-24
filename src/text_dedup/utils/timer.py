import time
from types import TracebackType
from typing import Any

from loguru import logger


class TimerAttrError(Exception):
    def __init__(self, var_name: str) -> None:
        super().__init__(f"Variable {var_name} is not set")


class TimerContext:
    def __init__(self, timer: "Timer", name: str):
        self.timer = timer
        self.name = name
        self.start_time: float | None = None

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_value is not None:
            raise exc_value
        if self.start_time is None:
            raise TimerAttrError("start_time")
        self.timer.elapsed_times[self.name] = time.perf_counter() - self.start_time


class Timer:
    def __init__(self) -> None:
        self.elapsed_times: dict[str, float] = {}
        self._pad = 0

    def __call__(self, name: str) -> TimerContext:
        self._pad = max(self._pad, len(name))
        return TimerContext(self, name)

    def report(self, additional_info: dict[str, Any] | None = None) -> None:
        if additional_info:
            self._pad = max(self._pad, max(len(key) for key in additional_info))

        for name, elapsed_time in self.elapsed_times.items():
            logger.info(f"{name:>{self._pad + 2}}: {elapsed_time:.2f}s")

        if additional_info is not None:
            for key, value in additional_info.items():
                logger.info(f"{key:>{self._pad + 2}}: {value}")

    @property
    def pad(self) -> int:
        return self._pad + 2
