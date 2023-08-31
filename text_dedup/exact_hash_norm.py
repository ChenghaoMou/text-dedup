#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
import gc
import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import load_dataset
from text_dedup.utils import add_exact_hash_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils import deep_update
from text_dedup.utils.hashfunc import md5_hexdigest
from text_dedup.utils.hashfunc import sha256_hexdigest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.timer import Timer
from text_dedup.utils.preprocess import normalize as normalize_for_dedup, normalize_new_lines

def compute_hash(example: Dict[str, Any], idx: int, column: str, hash_func: Callable) -> Dict[str, Any]:
    """
    Compute a hash for each line in the document.

    Parameters
    ----------
    example : Dict[str, Any]
        One example.
    idx : List[int]
        The index of the example in the dataset.
    column : str
        The column name of the text.
    hash_func : Callable
        The hash function to use.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the hashes, the index of the example, and the index of the lines.
    """
    doc = normalize_for_dedup(normalize_new_lines(example[column]))
    hash = hash_func(bytes(doc, encoding="utf-8"))
    return {
        "__hash__": hash,
        "__idx__": idx
    }


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="text_dedup.exact_hash_norm",
        description="Deduplicate text using exact hashing with normalization",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_exact_hash_args(parser)
    args = parser.parse_args()

    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds: Dataset = load_dataset(args)

        # we use the hex digests for md5 and sha256 for legacy compatibility reasons
        # we use the raw xxh3_128 byte digests for speed
        hash_func: Callable = {
            "md5": md5_hexdigest,  # type: ignore
            "sha256": sha256_hexdigest,  # type: ignore
            "xxh3": xxh3_128_digest,  # type: ignore
        }[args.hash_func]

        LEN_DATASET: int = len(ds)
        hashes = {}

        with timer("Hashing"):
            ds = ds.map(
                compute_hash,
                batched=False,
                with_indices=True,
                num_proc=args.num_workers,
                fn_kwargs={"column": args.column, "hash_func": hash_func},
                desc="Computing hashes...",
            )

            # currently processing is done on a single thread.
            # still, due to the nature of the calculations it is O(len(ds))
            # to make multithreaded, would have to handle shared data structs etc.
            # most approaches are not low hanging fruit..
            NUM_SHARDS = int(np.ceil(LEN_DATASET / args.batch_size))
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = (
                    ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                    # TODO .map(either preprocessing like example.encode("utf-8") or multithreaded)
                )
                for i, h in tqdm(
                    zip(ds_shard["__idx__"], ds_shard["__hash__"]),
                    leave=False,
                ):
                    if h not in hashes:
                        #hashes[hash] = [main_index, cluster_size]
                        hashes[h] = [i, 1]
                    else:
                        #up 1 on cluster_size
                        hashes[h][1] = hashes[h][1] + 1

        with timer("Processing"):
            # gc manipulations to ensure that hashes object is not unneccessarily copied across processes
            gc.freeze()
            gc.disable()
            def mapping(record, idx):
                cluster_id, cluster_size = hashes[record['__hash__']]
                meta = {
                    'meta': {
                        'dedup': {
                            'exact_norm': {
                                'exact_hash_idx': idx,
                                'cluster_main_idx': cluster_id,
                                'cluster_size': cluster_size,
                                'is_duplicate': cluster_id != idx
                            }
                        }
                    }
                }
                return deep_update(record, meta)

            ds = ds.map(
                function=mapping,
                with_indices=True,
                num_proc=args.num_workers,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding duplicates..."
            )
            gc.enable()
            gc.collect()

            ds = ds.remove_columns(["__idx__", "__hash__"])
        if args.filter:
            with timer("Filtering"):
                # batch size here would be a trade off between memory and speed
                # default is 1000
                ds_final = ds.filter(
                    lambda record: not record['meta']['dedup']["exact_norm"]['is_duplicate'],
                    num_proc=args.num_workers,
                    writer_batch_size=args.batch_size,
                )
        else:
            ds_final = ds


        with timer("Saving"):
            if args.filter and args.save_both:
                ds.save_to_disk(args.output+"_orig")
            ds_final.save_to_disk(args.output)

        with timer("Cleaning"):
            if args.clean_cache:
                ds_final.cleanup_cache_files()
                ds.cleanup_cache_files()

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(ds_final)}")
