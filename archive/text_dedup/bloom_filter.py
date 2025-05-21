#!/usr/bin/env python
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import multiprocessing as mp
from typing import Callable

import click
import numpy as np
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import BloomFilterArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.load import load_hf_dataset
from text_dedup.utils.memory import DisableReferenceCount
from text_dedup.utils.timer import Timer

mp.set_start_method("fork", force=True)


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@BloomFilterArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    bloom_filter_args: BloomFilterArgs,
):
    timer = Timer()
    flags = []

    hash_func: Callable = {
        "md5": md5_digest,  # type: ignore
        "sha256": sha256_digest,  # type: ignore
        "xxh3": xxh3_128_digest,  # type: ignore
    }[bloom_filter_args.hash_func]

    bf = ScalableBloomFilter(
        initial_capacity=bloom_filter_args.initial_capacity,
        mode=ScalableBloomFilter.SMALL_SET_GROWTH,
        error_rate=bloom_filter_args.error_rate,
    )

    with timer("Total"):
        with timer("Loading"):
            ds, _ = load_hf_dataset(io_args=io_args, meta_args=meta_args)

        LEN_DATASET = len(ds)
        NUM_SHARDS = int(np.ceil(LEN_DATASET / meta_args.batch_size))

        with timer("Processing"):
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                # TODO .map(either preprocessing like example.encode("utf-8") or multithreaded)
                ds_shard = ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                for example in tqdm(ds_shard[meta_args.column], leave=False):
                    h = hash_func(example.encode("utf-8"))
                    # True if the element is seen, False otherwise
                    flags.append(bf.add(h))

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=io_args.num_proc,
                desc="Filtering...",
            )

        with timer("Saving"):
            ds.save_to_disk(io_args.output)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
