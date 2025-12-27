"""
Unified benchmark runner script.

This script runs benchmarks on different datasets with different algorithms.
It's designed to be called from the justfile.
"""

from pathlib import Path

import typer
from datasets import Dataset
from datasets import disable_progress_bars
from datasets import enable_progress_bars
from datasets import load_dataset

from benchmarks.benchmark_core import prepare_ground_truth as prepare_core_gt
from benchmarks.benchmark_core import print_results as print_core_results
from benchmarks.benchmark_core import run_minhash_benchmark as run_core_minhash
from benchmarks.benchmark_core import run_simhash_benchmark as run_core_simhash
from benchmarks.benchmark_news import prepare_ground_truth as prepare_news_gt
from benchmarks.benchmark_news import print_results as print_news_results
from benchmarks.benchmark_news import run_minhash_benchmark as run_news_minhash
from benchmarks.benchmark_news import run_simhash_benchmark as run_news_simhash
from text_dedup.config.base import load_config_from_toml
from text_dedup.utils.preprocess import news_copy_preprocessing


def run_core_benchmarks(algorithms: list[str]) -> dict[str, tuple[dict, float]]:
    """Run benchmarks on CORE dataset.

    Parameters
    ----------
    algorithms : list[str]
        List of algorithms to run: "minhash", "simhash"

    Returns
    -------
    dict
        Results mapping algorithm name to (metrics, time)
    """
    disable_progress_bars()
    dataset: Dataset = load_dataset("pinecone/core-2020-05-10-deduplication", split="train")  # pyright: ignore[reportAssignmentType]
    dataset = dataset.map(lambda x: {"text": " ".join((x["processed_title"], x["processed_abstract"])).lower()})
    id_to_core_id, labels = prepare_core_gt(dataset)
    enable_progress_bars()

    results = {}

    if "minhash" in algorithms:
        config_path = Path("configs/benchmark_core_minhash.toml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")  # noqa: TRY003

        config = load_config_from_toml(config_path)
        metrics, elapsed = run_core_minhash(config, labels, id_to_core_id, dataset=dataset)
        results["MinHash"] = (metrics, elapsed)

    if "simhash" in algorithms:
        config_path = Path("configs/benchmark_core_simhash.toml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")  # noqa: TRY003

        config = load_config_from_toml(config_path)
        metrics, elapsed = run_core_simhash(config, labels, id_to_core_id, dataset=dataset)
        results["SimHash"] = (metrics, elapsed)

    return results


def run_news_benchmarks(algorithms: list[str]) -> dict[str, tuple[float, float]]:
    """Run benchmarks on NEWS-COPY dataset.

    Parameters
    ----------
    algorithms : list[str]
        List of algorithms to run: "minhash", "simhash"

    Returns
    -------
    dict
        Results mapping algorithm name to (ARI, time)
    """
    disable_progress_bars()
    dataset: Dataset = load_dataset("chenghao/NEWS-COPY-eval", split="test")  # pyright: ignore[reportAssignmentType]
    dataset = dataset.map(lambda x: {"text": news_copy_preprocessing(x["article"])})  # pyright: ignore[reportAssignmentType]
    ground_truth = prepare_news_gt(dataset)
    enable_progress_bars()

    results = {}

    if "minhash" in algorithms:
        config_path = Path("configs/benchmark_news_minhash.toml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")  # noqa: TRY003

        config = load_config_from_toml(config_path)
        ari, elapsed = run_news_minhash(config, ground_truth, dataset=dataset)
        results["MinHash"] = (ari, elapsed)

    if "simhash" in algorithms:
        config_path = Path("configs/benchmark_news_simhash.toml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")  # noqa: TRY003

        config = load_config_from_toml(config_path)
        ari, elapsed = run_news_simhash(config, ground_truth, dataset=dataset)
        results["SimHash"] = (ari, elapsed)

    return results


def main(
    dataset: str = typer.Option("all", help="Which dataset to benchmark (default: all)"),
    algorithms: str = typer.Option("all", help="Which algorithms to run (default: all)"),
) -> None:
    if dataset not in ["core", "news", "all"]:
        raise ValueError(f"Invalid dataset: {dataset}")  # noqa: TRY003
    datasets = ["core", "news"] if dataset == "all" else [dataset]
    algorithms_list = ["minhash", "simhash"] if algorithms == "all" else algorithms.split(",")
    for algo in algorithms_list:
        if algo not in ["minhash", "simhash"]:
            raise ValueError(f"Invalid algorithm: {algo}")  # noqa: TRY003

    if "core" in datasets:
        core_results = run_core_benchmarks(algorithms_list)
        print_core_results(core_results)

    if "news" in datasets:
        news_results = run_news_benchmarks(algorithms_list)
        print_news_results(news_results)


if __name__ == "__main__":
    typer.run(main)
