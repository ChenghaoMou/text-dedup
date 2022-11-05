#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
import time
from hashlib import md5

from datasets import load_dataset
from pybloom_live import ScalableBloomFilter

from text_dedup import logger
from text_dedup.utils import add_bloom_filter_args, add_io_args, add_meta_args

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="text_dedup.bloomfilter",
        description="Deduplicate text using Bloom Filter",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_bloom_filter_args(parser)
    args = parser.parse_args()

    if args.path is not None:
        elapsed_time = {"All": time.time()}
        # region: loading
        elapsed_time["Loading"] = time.time()
        ds = load_dataset(
            path=args.path,
            name=args.name,
            data_dir=args.data_dir,
            data_files=args.data_files,
            split=args.split,
            cache_dir=args.cache_dir,
            use_auth_token=args.use_auth_token,
        )
        elapsed_time["Loading"] = time.time() - elapsed_time["Loading"]
        # endregion

        hash_func = {
            "md5": md5,
        }[args.hash_func]
        bf = ScalableBloomFilter(
            initial_capacity=args.initial_capacity,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH,
            error_rate=args.error_rate,
        )
        flags = []

        # region: processing
        elapsed_time["Processing"] = time.time()
        for example in ds:
            h = hash_func(example[args.column].encode("utf-8")).hexdigest()
            # True if the element is seen, False otherwise
            flags.append(bf.add(h))
        elapsed_time["Processing"] = time.time() - elapsed_time["Processing"]
        # endregion

        # region: filtering and save
        elapsed_time["Filtering"] = time.time()
        ds = ds.filter(lambda _, idx: not flags[idx], with_indices=True, num_proc=os.cpu_count())
        elapsed_time["Filtering"] = time.time() - elapsed_time["Filtering"]
        elapsed_time["Saving"] = time.time()
        ds.save_to_disk(os.path.join(args.output_dir, args.dedup_name))
        elapsed_time["Saving"] = time.time() - elapsed_time["Saving"]
        elapsed_time["All"] = time.time() - elapsed_time["All"]
        # endregion

        for k, v in elapsed_time.items():
            logger.info(f"{k:<30}: {v:.2f}s")

        logger.info(f"{'Before':<30}: {len(flags)}")
        logger.info(f"{'After':<30}: {len(ds)}")
