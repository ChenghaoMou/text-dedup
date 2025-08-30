from .logger import log


def check_env():
    import polars as pl
    import psutil

    log.info(f"Polars thread pool size: {pl.thread_pool_size()}")
    log.info(f"CPU count: {psutil.cpu_count()}")
    log.info(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
