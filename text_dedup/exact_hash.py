#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
from typing import Callable

from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_exact_hash_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils.hashfunc import md5
from text_dedup.utils.hashfunc import sha256
from text_dedup.utils.hashfunc import xxh3_128
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

    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds: Dataset = load_dataset(  # type: ignore
                path=args.path,
                name=args.name,
                data_dir=args.data_dir,
                data_files=args.data_files,
                split=args.split,
                revision=args.revision,
                cache_dir=args.cache_dir,
                token=args.use_auth_token,
            )

        hash_func: Callable = {
            "md5": md5,  # type: ignore
            "sha256": sha256,  # type: ignore
            "xxh3": xxh3_128,  # type: ignore
        }[args.hash_func]

        hashes = set()
        flags = []

        with timer("Processing"):
            for idx in tqdm(range(0, len(ds), args.batch_size), desc="Processing..."):
                batch = ds[idx : idx + args.batch_size]
                for example in tqdm(batch[args.column], leave=False):
                    h = hash_func(example.encode("utf-8")).hexdigest()
                    if h in hashes:
                        flags.append(True)
                    else:
                        flags.append(False)
                        hashes.add(h)

        with timer("Filtering"):
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=os.cpu_count(),
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
