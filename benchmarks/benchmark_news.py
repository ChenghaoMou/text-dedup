"""
Benchmark on NEWS-COPY dataset.

This benchmark evaluates deduplication algorithms on the NEWS-COPY dataset,
which contains news articles with cluster labels for near-duplicate detection.
Uses Adjusted Rand Index (ARI) as the evaluation metric.
"""

import pickle
from pathlib import Path

from datasets import disable_progress_bars
from huggingface_hub.utils.tqdm import enable_progress_bars
from rich.console import Console
from rich.table import Table
from sklearn.metrics import adjusted_rand_score

from text_dedup.config import Config
from text_dedup.minhash import main as minhash_main
from text_dedup.simhash import main as simhash_main
from text_dedup.utils.timer import Timer


def prepare_ground_truth(dataset):
    """Prepare ground truth cluster labels from the dataset.

    Parameters
    ----------
    dataset
        HuggingFace dataset with 'cluster' field

    Returns
    -------
    list
        List of cluster labels, one per document
    """
    return [record["cluster"] for record in dataset]


def evaluate_clustering(ground_truth: list[int], predictions: dict[int, int]) -> float:
    """Evaluate clustering using Adjusted Rand Index.

    Parameters
    ----------
    ground_truth : list[int]
        True cluster labels
    predictions : dict[int, int]
        Predicted cluster assignments (index -> cluster_id)

    Returns
    -------
    float
        Adjusted Rand Index score
    """
    # Convert predictions dict to list aligned with ground_truth
    pred_labels = [predictions.get(i, i) for i in range(len(ground_truth))]
    return adjusted_rand_score(ground_truth, pred_labels)


def run_minhash_benchmark(config: Config, ground_truth: list[int], dataset=None) -> tuple[float, float]:
    """Run MinHash benchmark.

    Parameters
    ----------
    config : Config
        MinHash configuration
    ground_truth : list[int]
        Ground truth cluster labels
    dataset : Dataset, optional
        Preprocessed dataset to use instead of loading from config

    Returns
    -------
    tuple
        (ARI score, elapsed_time)
    """
    timer = Timer()
    with timer("MinHash"):
        if dataset is not None:
            disable_progress_bars()
            temp_path = Path(config.output.output_dir) / "temp_dataset"
            temp_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(temp_path))

            # Update config to load from temp location
            original_read_args = config.input.read_arguments.copy()
            config.input.read_arguments = {"dataset_path": str(temp_path)}
            enable_progress_bars()

            minhash_main(config)

            # Cleanup
            import shutil

            shutil.rmtree(temp_path, ignore_errors=True)
            config.input.read_arguments = original_read_args
        else:
            minhash_main(config)

    # Load cluster results
    output_dir = Path(config.output.output_dir)
    if config.output.save_clusters:
        with open(output_dir / "clusters.pickle", "rb") as f:
            cluster_mapping = pickle.load(f)  # noqa: S301
    else:
        # If no clusters saved, assume each document is its own cluster
        cluster_mapping = {i: i for i in range(len(ground_truth))}

    ari = evaluate_clustering(ground_truth, cluster_mapping)
    return ari, timer.elapsed_times["MinHash"]


def run_simhash_benchmark(config: Config, ground_truth: list[int], dataset=None) -> tuple[float, float]:
    """Run SimHash benchmark.

    Parameters
    ----------
    config : Config
        SimHash configuration
    ground_truth : list[int]
        Ground truth cluster labels
    dataset : Dataset, optional
        Preprocessed dataset to use instead of loading from config

    Returns
    -------
    tuple
        (ARI score, elapsed_time)
    """
    timer = Timer()
    with timer("SimHash"):
        if dataset is not None:
            # Save preprocessed dataset temporarily
            disable_progress_bars()
            temp_path = Path(config.output.output_dir) / "temp_dataset"
            temp_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(temp_path))

            # Update config to load from temp location
            original_read_args = config.input.read_arguments.copy()
            config.input.read_arguments = {"dataset_path": str(temp_path)}
            enable_progress_bars()
            simhash_main(config)

            # Cleanup
            import shutil

            shutil.rmtree(temp_path, ignore_errors=True)
            config.input.read_arguments = original_read_args
        else:
            simhash_main(config)

    # Load cluster results
    output_dir = Path(config.output.output_dir)
    if config.output.save_clusters:
        with open(output_dir / "clusters.pickle", "rb") as f:
            cluster_mapping = pickle.load(f)  # noqa: S301
    else:
        # If no clusters saved, assume each document is its own cluster
        cluster_mapping = {i: i for i in range(len(ground_truth))}

    ari = evaluate_clustering(ground_truth, cluster_mapping)
    return ari, timer.elapsed_times["SimHash"]


def print_results(results: dict[str, tuple[float, float]]) -> None:
    """Print benchmark results in a formatted table.

    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to (ARI, time) tuple
    """
    console = Console()

    table = Table(title="Benchmark Results on NEWS-COPY Dataset")
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("ARI Score", justify="right")
    table.add_column("Time", justify="right")

    for algo_name, (ari, elapsed) in results.items():
        table.add_row(
            algo_name,
            f"{ari:.4f}",
            f"{elapsed:.2f}s",
        )

    console.print()
    console.print(table)
