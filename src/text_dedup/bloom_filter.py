import uuid
from typing import cast

from datasets import Dataset
from rbloom import Bloom

from text_dedup.config import BloomFilterAlgorithmConfig
from text_dedup.config import Config
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.logger import log
from text_dedup.utils.timer import Timer


def load_and_preprocess(config: Config) -> tuple[Dataset, int]:
    """Load and preprocess the dataset."""
    ds = load_dataset(config)
    original_len = len(ds)
    return ds, original_len


def bloom_filter(config: Config, ds: Dataset) -> Dataset:
    """Bloom filter the dataset."""
    algo = cast(BloomFilterAlgorithmConfig, config.algorithm)
    bf: Bloom = algo.get_filter()

    if config.algorithm.num_proc > 1:
        log.warning(
            "Bloom filter does not support multi-processing due to state requirements. Using num_proc=1 instead."
        )

    def f(text: str) -> dict[str, bool]:
        if text in bf:
            return {"duplicate": True}
        bf.add(text)
        return {"duplicate": False}

    ds = ds.map(
        f,
        num_proc=1,
        desc="Indexing...",
        input_columns=[algo.text_column],
        # * Bloom object is not pickleable
        new_fingerprint=str(uuid.uuid4()),
    )
    return ds


def remove_duplicates(config: Config, ds: Dataset) -> Dataset:
    """Remove duplicates from the dataset."""
    if not config.output.skip_filtering:
        result: Dataset = ds.filter(
            function=lambda record: not record["duplicate"],  # pyright: ignore[reportUnknownLambdaType]
            with_indices=False,
            num_proc=config.algorithm.num_proc,
            desc="Removing duplicates...",
        )
        return result
    return ds


def main(config: Config) -> None:
    """
    Running BloomFilter algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    timer = Timer()

    with timer("Total", enable_spin=False):
        with timer("Preprocessing", enable_spin=False):
            ds, ORIGINAL_LEN = load_and_preprocess(config)

        with timer("Indexing", enable_spin=False):
            ds = bloom_filter(config, ds)

        with timer("Filtering", enable_spin=False):
            final_data = remove_duplicates(config, ds)

        with timer("Saving"):
            save_dataset(config, final_data=final_data, clusters={})

        with timer("Cleaning"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    timer.report({"Before": ORIGINAL_LEN, "After": len(final_data)})


if __name__ == "__main__":
    from pydantic_settings import CliApp

    from text_dedup.config.base import Config
    from text_dedup.utils.env import check_env
    from text_dedup.utils.progress import use_custom_progress_bar

    with use_custom_progress_bar():
        config = CliApp.run(Config)
        check_env()
        if config.debug.enable_profiling:
            from scalene.scalene_profiler import enable_profiling

            with enable_profiling():
                main(config)
        else:
            main(config)
