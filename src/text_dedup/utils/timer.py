import time
from types import TracebackType
from typing import Any

from rich.progress import Progress
from rich.progress import TaskID

from .logger import log


class TimerAttrError(Exception):
    def __init__(self, var_name: str) -> None:
        super().__init__(f"Variable {var_name} is not set")


class TimerContext:
    def __init__(self, timer: "Timer", name: str, enable_spin: bool = True):
        self.timer: Timer = timer
        self.name: str = name
        self.start_time: float | None = None
        self.enable_spin: bool = enable_spin
        self.spinner: Progress = Progress(transient=True)
        self.task: TaskID | None = None

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()
        if self.enable_spin:
            self.spinner.start()
            self.task = self.spinner.add_task(self.name, total=None)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_value is not None:
            raise exc_value
        if self.start_time is None:
            raise TimerAttrError("start_time")
        self.timer.elapsed_times[self.name] = time.perf_counter() - self.start_time
        if self.enable_spin:
            self.spinner.stop()


class Timer:
    def __init__(self) -> None:
        self.elapsed_times: dict[str, float] = {}
        self._pad: int = 0

    def __call__(self, name: str, enable_spin: bool = True) -> TimerContext:
        self._pad = max(self._pad, len(name))
        return TimerContext(self, name, enable_spin=enable_spin)

    def _is_total(self, name: str) -> bool:
        return name.lower() in {"total", "total time"}

    def report(self, additional_info: dict[str, Any] | None = None) -> None:  # pyright: ignore[reportExplicitAny]
        if additional_info:
            self._pad = max(self._pad, max(len(key) for key in additional_info))

        time_pad = max(len(f"{elapsed_time:.2f}") for elapsed_time in self.elapsed_times.values())

        total_time = sum(v for k, v in self.elapsed_times.items() if not self._is_total(k))

        for name, elapsed_time in self.elapsed_times.items():
            percentage = "" if self._is_total(name) else f"({elapsed_time / total_time * 100:>5.2f}%)"
            log.info(
                f"[green]{name:>{self._pad + 2}}[/green]: {f'{elapsed_time:.2f} s':<{time_pad + 1}} {percentage}",
                extra={"markup": True},
            )

        if additional_info is not None:
            for key, value in additional_info.items():  # pyright: ignore[reportAny]
                log.info(f"[green]{key:>{self._pad + 2}}[/green]: {value}", extra={"markup": True})

    @property
    def pad(self) -> int:
        return self._pad + 2
