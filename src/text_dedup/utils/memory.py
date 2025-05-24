import gc
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def disable_reference_count() -> Generator[None, None, None]:
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()
