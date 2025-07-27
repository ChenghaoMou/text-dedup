from collections import Counter
from collections import defaultdict
from typing import Any
from typing import cast

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from text_dedup.config import Config
from text_dedup.config import MinHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.cache import config_fingerprint
from text_dedup.utils.jaccard import jaccard_similarity
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find_rust import UnionFind


def load_and_preprocess(config: Config) -> Dataset:
    """Load and preprocess the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ds = load_dataset(config)
    result: Dataset = ds.filter(
        minhash_args.get_filtering_func(),
        num_proc=config.algorithm.num_proc,
        new_fingerprint=config_fingerprint(config.input, suffix="preprocessing"),
    )
    return result


def embed(config: Config, ds: Dataset) -> Dataset:
    """Fingerprint the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    result: Dataset = ds.map(
        function=minhash_args.get_embed_func(),
        input_columns=[minhash_args.text_column, minhash_args.internal_index_column],
        remove_columns=[col for col in ds.column_names if col != minhash_args.internal_index_column],
        num_proc=config.algorithm.num_proc,
        desc="Fingerprinting...",
        load_from_cache_file=True,
        new_fingerprint=config_fingerprint(config.algorithm, suffix="hashing"),
    )
    return result


def clustering(config: Config, ds: Dataset) -> dict[int, int]:
    """Cluster the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    HASH_TABLES: list[dict[int, set]] = minhash_args.create_hash_tables()
    uf: UnionFind[int] = UnionFind[int](max_size=len(ds))

    def collect(batch: dict[str, list[Any]]) -> None:
        """Collect hash values from the fingerprints."""
        for key, Hs in zip(batch[minhash_args.internal_index_column], batch[minhash_args.signature_column]):
            for i, H in enumerate(Hs):
                HASH_TABLES[i][H].add(key)

    ds.map(
        function=collect,
        with_indices=False,
        num_proc=1,
        desc="Collecting...",
        batched=True,
        batch_size=config.algorithm.batch_size,
    )

    for table in tqdm(HASH_TABLES, desc="Clustering...", dynamic_ncols=True):
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                uf.union(x, idx)

    return {idx: uf.find(idx) for idx in ds[minhash_args.internal_index_column]}


def assign_clusters(config: Config, ds: Dataset, parents: dict[int, int]) -> Dataset:
    """Assign cluster id to the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ds = ds.map(
        function=lambda record: {minhash_args.cluster_column: parents[record[minhash_args.internal_index_column]]},
        with_indices=False,
        # ! parents is pickled to multiple processes
        num_proc=config.algorithm.num_proc,
        desc="Assigning initial clusters...",
    )
    return ds


def check_cluster(config: Config, texts_and_indices: list[tuple[str, int]]) -> tuple[set[int], set[int]]:
    """Find true positives and false positives from a cluster."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ngram_tokenizer = minhash_args.get_ngrams_func()

    true_positives: set[int] = set()
    false_positives: set[int] = set()

    for i, (text_i, idx_i) in enumerate(texts_and_indices):
        ngrams = ngram_tokenizer(text_i)
        for j, (text_j, idx_j) in enumerate(texts_and_indices):
            if i >= j:
                continue
            similarity = jaccard_similarity(
                ngrams,
                ngram_tokenizer(text_j),
            )
            if similarity >= minhash_args.threshold:
                true_positives.add(idx_i)
                true_positives.add(idx_j)

        if idx_i not in true_positives:
            false_positives.add(idx_i)

    return true_positives, false_positives


def retrieve_clusters(config: Config, ds: Dataset) -> dict[str, list[tuple[str, int]]]:
    """Retrieve clusters from a dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    cluster_sizes: dict[int, int] = Counter()

    def count_clusters(batch: dict[str, list[Any]]) -> None:
        """Count the size of each cluster."""
        for cluster_id in batch[minhash_args.cluster_column]:
            cluster_sizes[cluster_id] += 1

    ds.map(
        function=count_clusters,
        batched=True,
        batch_size=config.algorithm.batch_size,
        num_proc=1,
        desc="Counting cluster sizes...",
    )

    multi_item_clusters = {cid for cid, size in cluster_sizes.items() if size > 1}
    cluster_texts: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def collect_multi_cluster_texts(batch: dict[str, list[Any]]) -> None:
        """Collect text from each cluster."""
        for cluster_id, text, idx in zip(
            batch[minhash_args.cluster_column],
            batch[minhash_args.text_column],
            batch[minhash_args.internal_index_column],
        ):
            if cluster_id in multi_item_clusters:
                cluster_texts[cluster_id].append((text, idx))

    ds.map(
        function=collect_multi_cluster_texts,
        batched=True,
        batch_size=config.algorithm.batch_size,
        num_proc=1,
        desc="Collecting multi-item cluster texts...",
    )
    return cluster_texts


def check_false_positives(config: Config, ds: Dataset) -> tuple[Dataset, dict[int, int]]:
    """Check false positives."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    cluster_texts: dict[str, list[tuple[str, int]]] = retrieve_clusters(config, ds)
    new_parents: dict[int, int] = {}

    total = 0
    total_false_positives = 0
    total_true_positives = 0
    total_true_positive_clusters = 0

    for _, texts_and_indices in tqdm(cluster_texts.items(), desc="Verifying..."):
        true_positives, false_positives = check_cluster(
            config,
            texts_and_indices,
        )
        for idx in false_positives:
            new_parents[idx] = idx

        if true_positives:
            total_true_positive_clusters += 1
            new_cluster_id = min(true_positives)
            for idx in true_positives:
                new_parents[idx] = new_cluster_id

        total += len(texts_and_indices)
        total_false_positives += len(false_positives)
        total_true_positives += len(true_positives)

    logger.info(f"Verified {len(cluster_texts)} clusters")
    logger.info(f"False Positives   : {total_false_positives}")
    logger.info(f"True Positives    : {total_true_positives}")
    logger.info(f"True Clusters     : {total_true_positive_clusters}")

    ds = ds.map(
        function=lambda record: {
            minhash_args.cluster_column: new_parents.get(
                record[minhash_args.internal_index_column],
                record[minhash_args.internal_index_column],
            )
        },
        with_indices=False,
        # ! new_parents is pickled
        num_proc=config.algorithm.num_proc,
        desc="Updating clusters...",
    )

    return ds, new_parents


def filter_duplicates(config: Config, ds: Dataset) -> Dataset:
    """Remove duplicates from the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    if not config.output.skip_filtering:
        result: Dataset = ds.filter(
            function=lambda record: record[minhash_args.cluster_column] == record[minhash_args.internal_index_column],
            with_indices=False,
            num_proc=config.algorithm.num_proc,
            desc="Removing duplicates...",
        )
        return result
    return ds


def main(config: Config) -> None:
    """
    Running MinHash algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    timer = Timer()

    with timer("Total"):
        with timer("Preprocessing"):
            ds = load_and_preprocess(config)

        LEN_DATASET = len(ds)

        with timer("MinHashing"):
            embedded = embed(config, ds)

        with timer("Clustering"):
            PARENTS = clustering(config, embedded)

        with timer("Filtering"):
            ds = assign_clusters(config, ds, PARENTS)
            if minhash_args.check_false_positive:
                ds, PARENTS = check_false_positives(config, ds)
            final_data = filter_duplicates(config, ds)

        with timer("Saving"):
            save_dataset(config, final_data=final_data, clusters=PARENTS)

        with timer("Cleaning"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    timer.report({"Before": LEN_DATASET, "After": len(final_data)})


if __name__ == "__main__":
    from pydantic_settings import CliApp

    from text_dedup.config.base import Config

    main(CliApp.run(Config))
