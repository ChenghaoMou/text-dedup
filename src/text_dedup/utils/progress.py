from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from typing import Self

from rich.progress import Progress
from rich.progress import TaskID


class CustomProgressBar:
    def __init__(self, unit: str, total: int, initial: int = 0, desc: str = ""):
        self.unit: str = unit
        self.total: int = total
        self.initial: int = initial
        self.desc: str = desc
        self.progress: Progress = Progress(transient=True)
        self.task: TaskID | None = None

    def __enter__(self) -> Self:
        _ = self.progress.__enter__()
        self.task = self.progress.add_task(self.desc, total=self.total)
        self.update(self.initial)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.progress.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)  # type: ignore[call-arg]

    def update(self, n: int | float) -> None:
        if self.task is None:
            raise ValueError("Task is not set")  # noqa: TRY003
        self.progress.update(self.task, advance=n)

    def set_description(self, desc: str) -> None:
        self.desc = desc
        if self.task is not None:
            self.progress.update(self.task, description=desc)


def custom_track(unit: str, total: int, desc: str, initial: int = 0) -> CustomProgressBar:
    return CustomProgressBar(unit=unit, total=total, initial=initial, desc=desc)


@contextmanager
def use_custom_progress_bar() -> Generator[None, None, None]:
    from unittest.mock import patch

    with (
        patch("datasets.utils.tqdm", CustomProgressBar),
        patch("datasets.arrow_dataset.hf_tqdm", CustomProgressBar),
        # patch("datasets.arrow_reader.hf_tqdm", track),
    ):
        yield
