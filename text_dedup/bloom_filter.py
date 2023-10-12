#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
from typing import Callable

import datasets
import numpy as np
from datasets.load import load_dataset
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_bloom_filter_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.timer import Timer

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="text_dedup.bloomfilter",
        description="Deduplicate text using Bloom Filter",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_bloom_filter_args(parser)
    args = parser.parse_args()

    timer = Timer()
    flags = []

    with timer("Total"):
        with timer("Loading"):
            ds: datasets.Dataset = load_dataset(  # type: ignore
                path=args.path,
                name=args.name,
                data_dir=args.data_dir,
                data_files=args.data_files,
                split=args.split,
                revision=args.revision,
                cache_dir=args.cache_dir,
                token=args.use_auth_token,
                num_proc=args.num_proc,
            )

        hash_func: Callable = {
            "md5": md5_digest,  # type: ignore
            "sha256": sha256_digest,  # type: ignore
            "xxh3": xxh3_128_digest,  # type: ignore
        }[args.hash_func]

        LEN_DATASET = len(ds)

        bf = ScalableBloomFilter(
            initial_capacity=args.initial_capacity,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH,
            error_rate=args.error_rate,
        )
        with timer("Processing"):
            NUM_SHARDS = int(np.ceil(LEN_DATASET / args.batch_size))
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = (
                    ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                    # TODO .map(either preprocessing like example.encode("utf-8") or multithreaded)
                )
                for example in tqdm(ds_shard[args.column], leave=False):
                    h = hash_func(example.encode("utf-8"))
                    # True if the element is seen, False otherwise
                    flags.append(bf.add(h))

        with timer("Filtering"):
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=args.num_proc,
                desc="Filtering...",
            )

        with timer("Saving"):
            ds.save_to_disk(args.output)

        with timer("Cleaning"):
            if args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")
