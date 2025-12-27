"""
Benchmark on pinecone/core-2020-05-10-deduplication dataset.

This benchmark evaluates deduplication algorithms on the CORE dataset,
which contains academic papers with labeled duplicates.
"""

import pickle
from pathlib import Path

import pandas as pd
from datasets import Dataset
from datasets import disable_progress_bars
from datasets import enable_progress_bars
from rich.console import Console
from rich.table import Table

from benchmarks.utils import classify_prediction
from benchmarks.utils import clusters_to_predictions_minhash
from benchmarks.utils import clusters_to_predictions_simhash
from text_dedup.config import Config
from text_dedup.minhash import main as minhash_main
from text_dedup.simhash import main as simhash_main
from text_dedup.utils.timer import Timer


def prepare_ground_truth(dataset: Dataset) -> tuple[dict[int, str], dict[str, set[str]]]:
    """Prepare ground truth labels from the dataset.

    Parameters
    ----------
    dataset
        HuggingFace dataset with 'labelled_duplicates' field

    Returns
    -------
    tuple
        (id_to_core_id, labels) where labels is a dict mapping core_id to set of duplicate core_ids
    """
    id_to_core_id: dict[int, str] = {}
    labels: dict[str, set[str]] = {}

    for idx, record in enumerate(dataset):
        core_id = str(record["core_id"])  # pyright: ignore[reportUnknownArgumentType, reportCallIssue, reportArgumentType]
        id_to_core_id[idx] = core_id
        duplicates = set(record["labelled_duplicates"]) if record["labelled_duplicates"] else set()  # pyright: ignore[reportCallIssue, reportArgumentType]
        labels[core_id] = duplicates

    return id_to_core_id, labels


def evaluate_predictions(labels: dict[str, set[str]], predictions: dict[str, set[str]]) -> dict:
    """Evaluate predictions against ground truth labels.

    This follows the evaluation methodology from the reference benchmark:
    https://github.com/ChenghaoMou/text-dedup/blob/main/tests/benchmark_core.py

    Parameters
    ----------
    labels : dict[str, set[str]]
        Ground truth: mapping from ID to set of duplicate IDs
    predictions : dict[str, set[str]]
        Predictions: mapping from ID to set of duplicate IDs

    Returns
    -------
    dict
        Evaluation metrics
    """
    results: list[dict] = []
    for doc_id in labels:
        gt_dups = labels.get(doc_id, set())
        pred_dups = predictions.get(doc_id, set())

        results.append({
            "id": doc_id,
            "duplicates": gt_dups,
            "predictions": pred_dups,
            "classification": classify_prediction(gt_dups, pred_dups),
            "exact_match": gt_dups == pred_dups,
        })

    df = pd.DataFrame(results)
    accuracy = df["exact_match"].mean()

    # Get classification counts
    class_counts = df["classification"].value_counts().to_dict()
    tp = class_counts.get("TP", 0)
    tn = class_counts.get("TN", 0)
    fp = class_counts.get("FP", 0)
    fn = class_counts.get("FN", 0)

    # Calculate precision/recall for duplicates (Class)
    dup_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    dup_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate precision/recall for non-duplicates (Class_ / inverted)
    # After inversion: TP->TN, TN->TP, FP->FN, FN->FP
    # So for inverted: precision = TN / (TN + FN), recall = TN / (TN + FP)
    non_dup_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    non_dup_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Macro precision (average of duplicate and non-duplicate precision)
    # Note: The reference calls this "Macro F1 score" but it's actually macro precision
    macro_precision = (dup_precision + non_dup_precision) / 2

    return {
        "precision_duplicates": dup_precision,
        "recall_duplicates": dup_recall,
        "precision_non_duplicates": non_dup_precision,
        "recall_non_duplicates": non_dup_recall,
        "macro_f1": macro_precision,  # Keeping key name for compatibility
        "accuracy": accuracy,
        "class_distribution": class_counts,
    }


def run_minhash_benchmark(config: Config, labels: dict, id_to_core_id: dict, dataset=None) -> tuple[dict, float]:
    """Run MinHash benchmark.

    Parameters
    ----------
    config : Config
        MinHash configuration
    labels : dict
        Ground truth labels
    id_to_core_id : dict
        Mapping from internal index to core_id
    dataset : Dataset, optional
        Preprocessed dataset to use instead of loading from config

    Returns
    -------
    tuple
        (metrics, elapsed_time)
    """
    timer = Timer()
    with timer("MinHash", enable_spin=False):
        if dataset is not None:
            disable_progress_bars()
            temp_path = Path(config.output.output_dir) / "temp_dataset"
            temp_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(temp_path))

            original_read_args = config.input.read_arguments.copy()
            config.input.read_arguments = {"dataset_path": str(temp_path)}
            enable_progress_bars()
            minhash_main(config)

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

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)
    else:
        # If no clusters saved, assume no duplicates found
        predictions = {core_id: set() for core_id in labels}

    metrics = evaluate_predictions(labels, predictions)
    return metrics, timer.elapsed_times["MinHash"]


def run_simhash_benchmark(config: Config, labels: dict, id_to_core_id: dict, dataset=None) -> tuple[dict, float]:
    """Run SimHash benchmark.

    Parameters
    ----------
    config : Config
        SimHash configuration
    labels : dict
        Ground truth labels
    id_to_core_id : dict
        Mapping from internal index to core_id
    dataset : Dataset, optional
        Preprocessed dataset to use instead of loading from config

    Returns
    -------
    tuple
        (metrics, elapsed_time)
    """
    timer = Timer()
    with timer("SimHash", enable_spin=False):
        if dataset is not None:
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

        predictions: dict[str, set[str]] = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)
    else:
        # If no clusters saved, assume no duplicates found
        predictions = {core_id: set() for core_id in labels}

    metrics = evaluate_predictions(labels, predictions)
    return metrics, timer.elapsed_times["SimHash"]


def print_results(results: dict[str, tuple[dict, float]]) -> None:
    """Print benchmark results in a formatted table.

    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to (metrics, time) tuple
    """
    console = Console()

    table = Table(title="Benchmark Results on pinecone/core-2020-05-10-deduplication")
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("Precision (Duplicates)", justify="right")
    table.add_column("Recall (Duplicates)", justify="right")
    table.add_column("Precision (Non Duplicates)", justify="right")
    table.add_column("Recall (Non Duplicates)", justify="right")
    table.add_column("Macro F1 Score", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Time", justify="right")

    for algo_name, (metrics, elapsed) in results.items():
        table.add_row(
            algo_name,
            f"{metrics['precision_duplicates']:.4f}",
            f"{metrics['recall_duplicates']:.4f}",
            f"{metrics['precision_non_duplicates']:.4f}",
            f"{metrics['recall_non_duplicates']:.4f}",
            f"{metrics['macro_f1']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{elapsed:.2f}s",
        )

    console.print()
    console.print(table)
