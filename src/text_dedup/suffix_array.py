import shutil
from pathlib import Path
from typing import cast

from datasets import Dataset

from text_dedup.config import Config
from text_dedup.config import SuffixArrayAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.progress import use_custom_progress_bar
from text_dedup.utils.timer import Timer


def load_and_preprocess(config: Config) -> tuple[Dataset, int]:
    """Load and preprocess the dataset."""
    ds = load_dataset(config)
    original_size = ds.data.nbytes
    return ds, original_size


def main(config: Config) -> None:
    """
    Running Suffix Array algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    timer = Timer()
    algo = cast(SuffixArrayAlgorithmConfig, config.algorithm)
    cache_dir = Path(algo.google_repo_path) / algo.cache_dir
    temp_output_dir = Path(algo.google_repo_path) / "output"
    temp_dir = Path(algo.google_repo_path) / "tmp"
    temp_text = temp_output_dir / "temp_text.txt"
    temp_output = temp_output_dir / "temp_output.txt"

    for d in {cache_dir, temp_output_dir, temp_dir}:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(exist_ok=True, parents=True)

    with timer("Total", enable_spin=False), use_custom_progress_bar():
        with timer("Preprocessing", enable_spin=False):
            ds, ORIGINAL_SIZE = load_and_preprocess(config)
            offsets: list[slice] = []
            start = 0
            with open(temp_text, "wb") as f:
                for doc in ds:
                    doc_bytes = doc[algo.text_column].encode("utf-8")  # pyright: ignore[reportCallIssue, reportArgumentType]
                    end = start + len(doc_bytes)
                    offsets.append(slice(start, end))
                    start = end
                    f.write(doc_bytes)

        with timer("Making suffix array", enable_spin=True):
            algo.run_command(
                f"python scripts/make_suffix_array.py {temp_text.relative_to(algo.google_repo_path)}",
                algo.google_repo_path,
            )

        with timer("SelfSimilar", enable_spin=True):
            algo.run_command(
                f"cargo run self-similar --data-file {temp_text.relative_to(algo.google_repo_path)}"
                f" --length-threshold {algo.length_threshold} --cache-dir {cache_dir.relative_to(algo.google_repo_path)} --num-threads {algo.num_proc}",
                algo.google_repo_path,
            )
            algo.run_command(
                f"cargo run collect --data-file {temp_text.relative_to(algo.google_repo_path)}"
                f" --length-threshold {algo.length_threshold} --cache-dir {cache_dir.relative_to(algo.google_repo_path)} >"
                f" {temp_output.relative_to(algo.google_repo_path)}",
                algo.google_repo_path,
            )

        with timer("Restore", enable_spin=True):
            duplicate_slices, _ = algo.restore_and_merge(
                offsets,
                temp_output,
                algo.length_threshold,
                algo.merge_strategy,
            )

        with timer("Deduplicate"):
            ds = ds.map(
                lambda content, idx: {
                    algo.text_column: algo.clean_up(content, duplicate_slices[idx]),
                },
                with_indices=True,
                input_columns=[algo.text_column],
                desc="Deduplicating",
            ).filter(
                lambda content: len(content) > 0,
                input_columns=[algo.text_column],
                desc="Filtering empty documents",
            )

        with timer("Saving"):
            save_dataset(config, final_data=ds, clusters={})

        if config.output.clean_cache:
            with timer("Cleaning"):
                ds.cleanup_cache_files()
                shutil.rmtree(temp_output_dir)
                shutil.rmtree(temp_dir)
                shutil.rmtree(cache_dir)

    timer.report({"Before": f"{ORIGINAL_SIZE / 1024 / 1024:.2f} MB", "After": f"{ds.data.nbytes / 1024 / 1024:.2f} MB"})


if __name__ == "__main__":  # pragma: no cover
    from pydantic_settings import CliApp

    from text_dedup.utils.env import check_env

    config = CliApp.run(Config)
    check_env()
    if config.debug.enable_profiling:
        from scalene.scalene_profiler import enable_profiling

        with enable_profiling():
            main(config)
    else:
        main(config)
