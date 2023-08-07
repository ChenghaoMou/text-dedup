#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
from typing import Callable

from datasets import load_dataset
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_exact_hash_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils.hashfunc import md5_hexdigest
from text_dedup.utils.hashfunc import sha256_hexdigest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.timer import Timer

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="text_dedup.exacthash",
        description="Deduplicate text using exact hashing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_exact_hash_args(parser)
    args = parser.parse_args()

    NUM_PROC = os.cpu_count()
    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds = load_dataset(
                path=args.path,
                name=args.name,
                data_dir=args.data_dir,
                data_files=args.data_files,
                split=args.split,
                revision=args.revision,
                cache_dir=args.cache_dir,
                num_proc=NUM_PROC,
                token=args.use_auth_token,
            )

        # we use the hex digests for md5 and sha256 for legacy compatibility reasons
        # we use the raw xxh3_128 byte digests for speed
        hash_func: Callable = {
            "md5": md5_hexdigest,  # type: ignore
            "sha256": sha256_hexdigest,  # type: ignore
            "xxh3": xxh3_128_digest,  # type: ignore
        }[args.hash_func]

        hashes = set()
        flags = []

        with timer("Processing"):
            # currently processing is done on a single thread.
            # still, due to the nature of the calculations it is O(len(ds))
            # to make multithreaded, would have to handle shared data structs etc.
            # most approaches are not low hanging fruit.
            for idx in tqdm(range(0, len(ds), args.batch_size), desc="Processing..."):
                batch = ds[idx : idx + args.batch_size]
                for example in tqdm(batch[args.column], leave=False):
                    # moving this byte conversion outside the loop saw no improvement <1 GiB datasets
                    # might not be worth the added overhead
                    h = hash_func(example.encode("utf-8"))
                    if h in hashes:
                        flags.append(True)
                    else:
                        flags.append(False)
                        hashes.add(h)

        with timer("Filtering"):
            # batch size here would be a trade off between memory and speed
            # default is 1000
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=NUM_PROC,
                writer_batch_size=args.batch_size,
            )

        with timer("Saving"):
            ds.save_to_disk(args.output)
        ds.cleanup_cache_files()

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")
