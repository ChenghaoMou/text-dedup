#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
from hashlib import md5, sha256

from datasets.load import load_dataset
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_bloom_filter_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
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
            ds: Dataset = load_dataset(
                path=args.path,
                name=args.name,
                data_dir=args.data_dir,
                data_files=args.data_files,
                split=args.split,
                revision=args.revision,
                cache_dir=args.cache_dir,
                use_auth_token=args.use_auth_token,
                num_proc=os.cpu_count(),
            )

        hash_func = {
            "md5": md5,
            "sha256": sha256,
        }[args.hash_func]

        bf = ScalableBloomFilter(
            initial_capacity=args.initial_capacity,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH,
            error_rate=args.error_rate,
        )
        with timer("Processing"):
            for idx in tqdm(range(0, len(ds), args.batch_size), desc="Processing..."):
                batch = ds[idx : idx + args.batch_size]
                for example in tqdm(batch[args.column], leave=False):
                    h = hash_func(example.encode("utf-8")).hexdigest()
                    # True if the element is seen, False otherwise
                    flags.append(bf.add(h))

        with timer("Filtering"):
            ds = ds.filter(
                lambda _, idx: not flags[idx], with_indices=True, num_proc=os.cpu_count(), desc="Filtering..."
            )

        with timer("Saving"):
            ds.save_to_disk(args.output)

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")
