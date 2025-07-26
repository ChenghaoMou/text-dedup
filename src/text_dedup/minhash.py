from collections import Counter
from collections import defaultdict
from typing import Any
from typing import cast

from loguru import logger
from tqdm import tqdm

from text_dedup.config import Config
from text_dedup.config import MinHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.cache import config_fingerprint
from text_dedup.utils.memory import disable_reference_count
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind


def main(config: Config) -> None:  # noqa: C901
    uf: UnionFind[int] = UnionFind[int]()
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    HASH_TABLES: list[dict[int, set]] = minhash_args.create_hash_tables()
    timer = Timer()

    with timer("Total"):
        with timer("Preprocessing"):
            ds = load_dataset(config)
            ds = ds.filter(
                minhash_args.get_filtering_func(),
                num_proc=config.algorithm.num_proc,
                new_fingerprint=config_fingerprint(config.input, suffix="preprocessing"),
            )

        LEN_DATASET = len(ds)

        with timer("MinHashing"):
            embedded = ds.map(
                function=minhash_args.get_embed_func(),
                input_columns=[minhash_args.text_column, minhash_args.internal_index_column],
                remove_columns=[col for col in ds.column_names if col != minhash_args.internal_index_column],
                num_proc=config.algorithm.num_proc,
                desc="Fingerprinting...",
                load_from_cache_file=True,
                new_fingerprint=config_fingerprint(config.algorithm, suffix="hashing"),
            )

        with timer("Clustering"):
            edges = []

            def collect(batch: dict[str, list[Any]]) -> None:
                for key, Hs in zip(batch[minhash_args.internal_index_column], batch[minhash_args.signature_column]):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            embedded.map(
                function=collect,
                with_indices=False,
                num_proc=1,
                desc="Collecting...",
                load_from_cache_file=True,
                batched=True,
                batch_size=config.algorithm.batch_size,
                new_fingerprint=config_fingerprint(config.algorithm, suffix="clustering"),
            )

            logger.info(f"Number of tables: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, desc="Clustering...", dynamic_ncols=True):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)

            logger.info(f"Number of edges: {len(set(edges))}")

        with timer("Filtering"), disable_reference_count():
            ds = ds.map(
                function=lambda record: {
                    minhash_args.cluster_column: uf.find(record[minhash_args.internal_index_column])
                },
                with_indices=False,
                num_proc=config.algorithm.num_proc,
                new_fingerprint=config_fingerprint(config.algorithm, suffix="filtering"),
                desc="Finding clusters...",
            )

            multi_item_clusters: set[str] = set()
            valid_indices: set[int] = set()

            if minhash_args.check_false_positive:
                cluster_sizes: dict[str, int] = Counter()

                def count_clusters(batch: dict[str, list[Any]]) -> None:
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
                logger.info(f"Found {len(multi_item_clusters)} multi-item clusters to verify")
                cluster_texts: dict[str, list[tuple[str, int]]] = defaultdict(list)

                def collect_multi_cluster_texts(batch: dict[str, list[Any]]) -> None:
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

                from text_dedup.utils.jaccard import jaccard_similarity

                ngram_tokenizer = minhash_args.get_ngrams_func()

                for _, texts_and_indices in tqdm(cluster_texts.items(), desc="Pairwise verification..."):
                    processed = set()  # Track which documents have been processed

                    for i, (text_i, idx_i) in enumerate(texts_and_indices):
                        if i in processed:
                            continue  # Skip if already processed as part of a similar pair

                        found_similar = False
                        ngrams = ngram_tokenizer(text_i)
                        for j, (text_j, idx_j) in enumerate(texts_and_indices):
                            if i == j or j in processed:
                                continue
                            similarity = jaccard_similarity(
                                ngrams,
                                ngram_tokenizer(text_j),
                            )
                            if similarity >= minhash_args.threshold:
                                if idx_i <= idx_j:
                                    valid_indices.add(idx_i)
                                else:
                                    valid_indices.add(idx_j)

                                processed.add(i)
                                processed.add(j)
                                found_similar = True
                                break

                        if not found_similar and i not in processed:
                            valid_indices.add(idx_i)
                            processed.add(i)

                logger.info(f"Pairwise verification: kept {len(valid_indices)} out of multi-item clusters")

                # Update cluster information in ds
                ds = ds.map(
                    function=lambda record: {
                        minhash_args.cluster_column: record[minhash_args.internal_index_column]
                        if record[minhash_args.internal_index_column] in valid_indices
                        else record[minhash_args.cluster_column]
                    },
                    with_indices=False,
                    num_proc=config.algorithm.num_proc,
                    new_fingerprint=config_fingerprint(config.algorithm, suffix="filtering"),
                    desc="Updating clusters...",
                )
                for valid_index in valid_indices:
                    uf.parent[valid_index] = valid_index

            # This is where the deduplication happens
            # Since there is no easy group by in `datasets`
            # I will use this filter for now
            if not config.output.skip_filtering:

                def filter_with_fp_check(record: dict[str, Any]) -> bool:
                    cluster_id = record[minhash_args.cluster_column]
                    idx = record[minhash_args.internal_index_column]

                    # Single-item clusters: keep representative (cluster_id == idx)
                    if not minhash_args.check_false_positive or cluster_id not in multi_item_clusters:
                        return bool(cluster_id == idx)

                    # Multi-item clusters: keep only verified indices
                    return idx in valid_indices

                final_data = ds.filter(
                    function=filter_with_fp_check,
                    with_indices=False,
                    num_proc=config.algorithm.num_proc,
                    desc="Filtering with false positive check...",
                )
            else:
                final_data = ds

        with timer("Saving"):
            save_dataset(config, final_data=final_data, uf=uf)

        with timer("Cleaning"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    timer.report({"Before": LEN_DATASET, "After": len(final_data)})


if __name__ == "__main__":
    import multiprocessing as mp

    from pydantic_settings import CliApp

    from text_dedup.config.base import Config

    # combined with reference counting disabled, this makes sure
    # the Union Find data structure is only copy-on-write
    mp.set_start_method("fork", force=True)
    main(CliApp.run(Config))
