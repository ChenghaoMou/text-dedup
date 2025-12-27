from collections import defaultdict
from typing import Any
from typing import cast

from bitarray import frozenbitarray
from datasets import Dataset
from rich.progress import track
from tqdm import tqdm

from text_dedup.config import Config
from text_dedup.config import SimHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.jaccard import jaccard_similarity
from text_dedup.utils.logger import log
from text_dedup.utils.progress import use_custom_progress_bar
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind


def load_and_preprocess(config: Config) -> tuple[Dataset, int]:
    """Load and preprocess the dataset."""
    ds = load_dataset(config)
    original_len = len(ds)
    return ds, original_len


def fingerprint(config: Config, ds: Dataset) -> Dataset:
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    embedded: Dataset = ds.map(
        function=algo.get_embed_func(),
        input_columns=[algo.text_column, algo.internal_index_column],
        remove_columns=[col for col in ds.column_names if col != algo.internal_index_column],
        num_proc=algo.num_proc,
        with_indices=False,
        batched=True,
        batch_size=1,
        desc="SimHashing...",
    )
    return embedded


def cluster(config: Config, ds: Dataset) -> dict[int, int]:
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    uf = UnionFind()
    buckets: dict[Any, list[tuple[int, frozenbitarray]]] = defaultdict(list)

    # Get the data columns we need
    indices = ds[algo.internal_index_column]
    keys_list = ds["__key__"]
    vals_list = ds["__val__"]

    for idx, key, sig_bytes in track(
        zip(indices, keys_list, vals_list, strict=True),
        description="Clustering...",
        total=len(ds),
        transient=True,
    ):
        sig = frozenbitarray(buffer=sig_bytes)
        key_tuple = tuple(key)

        for other_idx, other_sig in buckets[key_tuple]:
            if other_idx == idx:
                continue
            if (sig ^ other_sig).count(1) <= algo.bit_diff:
                uf.union(idx, other_idx)

        buckets[key_tuple].append((idx, sig))

    clusters = uf.get_clusters()
    return {k: v for k, v in clusters.items() if k != v}


def assign(config: Config, ds: Dataset, parents: dict[int, int]) -> Dataset:
    """Assign cluster id to the dataset."""
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    ds = ds.map(
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            algo.cluster_column: parents.get(
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
            ),
            "__duplicate__": record[algo.internal_index_column] in parents,
        },
        with_indices=False,
        # ! parents is pickled to multiple processes
        num_proc=config.algorithm.num_proc,
        desc="Assigning initial clusters...",
    )
    return ds


def check_false_positives(config: Config, ds: Dataset) -> tuple[Dataset, dict[int, int]]:
    """
    Check false positives using Jaccard similarity.

    This version avoids polars map_elements by doing the computation in Python directly.
    """
    algo = cast(SimHashAlgorithmConfig, config.algorithm)

    # Filter to only candidates (items marked as duplicates)
    ds_candidates: Dataset = ds.filter(
        function=lambda x: x["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
        num_proc=algo.num_proc,
    )

    if len(ds_candidates) == 0:
        return ds, {}

    # Group candidates by cluster
    cluster_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i in range(len(ds_candidates)):
        record = ds_candidates[i]
        cluster_id = record[algo.cluster_column]
        idx = record[algo.internal_index_column]
        text = record[algo.text_column]
        cluster_groups[cluster_id].append((idx, text))

    cluster_num = len(cluster_groups)
    tokenizer = algo.get_ngrams_func()

    # Verify pairs within each cluster
    verified_pairs = 0
    true_pairs: list[tuple[int, int, int]] = []  # (idx1, idx2, cluster_id)

    for cluster_id, members in tqdm(cluster_groups.items(), desc="Verifying clusters..."):
        if len(members) < 2:
            continue

        # Compare all pairs within this cluster
        for i, (idx1, text1) in enumerate(members):
            tokens1 = set(tokenizer(text1))
            for j in range(i + 1, len(members)):
                idx2, text2 = members[j]
                verified_pairs += 1

                tokens2 = set(tokenizer(text2))
                similarity = jaccard_similarity(tokens1, tokens2)

                if similarity >= algo.jaccard_threshold:
                    true_pairs.append((idx1, idx2, cluster_id))

    # Build new cluster assignments from verified pairs
    uf = UnionFind()
    for idx1, idx2, _ in true_pairs:
        uf.union(idx1, idx2)

    new_parents = uf.get_clusters()
    # Only keep non-trivial assignments
    new_parents = {k: v for k, v in new_parents.items() if k != v}

    total_true_positives = len(new_parents)
    total_false_positives = len(ds_candidates) - total_true_positives
    total_true_positive_clusters = len(set(new_parents.values()))

    log.info(f"Verified {cluster_num} clusters/{verified_pairs} pairs")
    log.info(f"False Positives   : {total_false_positives}")
    log.info(f"True Positives    : {total_true_positives}")
    log.info(f"True Clusters     : {total_true_positive_clusters}")

    ds = ds.map(
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            algo.cluster_column: new_parents.get(
                record[algo.internal_index_column],
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
            )
        },
        with_indices=False,
        # ! new_parents is pickled
        num_proc=config.algorithm.num_proc,
        desc="Updating clusters...",
    )

    return ds, new_parents


def remove_duplicates(config: Config, ds: Dataset) -> Dataset:
    """Remove duplicates from the dataset."""
    if not config.output.skip_filtering:
        result: Dataset = ds.filter(
            function=lambda record: not record["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
            with_indices=False,
            num_proc=config.algorithm.num_proc,
            desc="Removing duplicates...",
        )
        return result
    return ds


def main(config: Config) -> None:
    """
    Running SimHash algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    timer = Timer()
    algo = cast(SimHashAlgorithmConfig, config.algorithm)

    with timer("Total", enable_spin=False), use_custom_progress_bar():
        with timer("Preprocessing", enable_spin=False):
            ds, ORIGINAL_LEN = load_and_preprocess(config)

        with timer("SimHashing", enable_spin=False):
            embedded = fingerprint(config, ds)

        with timer("Clustering", enable_spin=False):
            assignment = cluster(config, embedded)
            ds = assign(config, ds, assignment)

        with timer("Verifying"):
            if algo.check_false_positive:
                ds, assignment = check_false_positives(config, ds)

        with timer("Filtering", enable_spin=False):
            final_data = remove_duplicates(config, ds)

        with timer("Saving"):
            save_dataset(config, final_data=final_data, clusters=assignment)

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

    config = CliApp.run(Config)
    check_env()
    if config.debug.enable_profiling:
        from scalene.scalene_profiler import enable_profiling

        with enable_profiling():
            main(config)
    else:
        main(config)
