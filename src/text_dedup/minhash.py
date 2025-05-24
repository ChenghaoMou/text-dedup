#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import multiprocessing as mp
import random
from typing import cast

import numpy as np
from loguru import logger
from tqdm import tqdm

from text_dedup.config.base import Config
from text_dedup.config.base import MinHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.memory import disable_reference_count
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind

# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)


def main(config: Config) -> None:
    uf: UnionFind[int] = UnionFind[int]()
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    HASH_TABLES: list[dict[int, set]] = minhash_args.hash_tables
    timer = Timer()

    with timer("Total"):
        with timer("Loading and preprocessing data"):
            ds = load_dataset(config)
            ds = ds.filter(
                minhash_args.get_filtering_func(),
                num_proc=config.algorithm.num_proc,
            )

        LEN_DATASET = len(ds)

        with timer("MinHashing text"):
            embedded = ds.map(
                function=minhash_args.get_embed_func(),
                input_columns=[minhash_args.text_column, minhash_args.internal_index_column],
                remove_columns=[col for col in ds.column_names if col != minhash_args.internal_index_column],
                num_proc=config.algorithm.num_proc,
                with_indices=False,
                desc="Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            NUM_SHARDS = np.ceil(LEN_EMBEDDED / config.algorithm.batch_size).astype(int)

        with timer("Clustering"):
            edges = []
            for i in tqdm(
                range(NUM_SHARDS),
                desc="Iterating MinHashes...",
                dynamic_ncols=True,
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=config.algorithm.batch_size,
                )
                for key, Hs in zip(
                    embedded_shard[minhash_args.internal_index_column], embedded_shard[minhash_args.signature_column]
                ):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            logger.info(f"Number of clusters: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, desc="Clustering...", dynamic_ncols=True):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)

            logger.info(f"Number of edges: {len(set(edges))}")

        with timer("Filtering clusters"), disable_reference_count():
            ds = ds.map(
                function=lambda record: {
                    minhash_args.cluster_column: uf.find(record[minhash_args.internal_index_column])
                },
                with_indices=False,
                num_proc=config.algorithm.num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            if not config.output.skip_filtering:
                final_data = ds.filter(
                    function=lambda record: record[minhash_args.cluster_column]
                    == record[minhash_args.internal_index_column],
                    with_indices=False,
                    num_proc=config.algorithm.num_proc,
                    desc="Filtering clusters...",
                )
            else:
                final_data = ds

        with timer("Saving data"):
            save_dataset(config, final_data=final_data, uf=uf)

        with timer("Cleaning cache"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    timer.report({"Before": LEN_DATASET, "After": len(final_data)})


if __name__ == "__main__":
    from pydantic_settings import CliApp

    from text_dedup.config.base import Config

    s = CliApp.run(Config)
    main(s)
