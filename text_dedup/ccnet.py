#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2023-05-06 19:34:35
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
# @Description  : Line-level deduplication based on Exact Hashing
# @Reference    : https://github.com/facebookresearch/cc_net/blob/main/cc_net/dedup.py

import argparse
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_exact_hash_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_64_digest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.preprocess import normalize as normalize_for_dedup
from text_dedup.utils.timer import Timer

HASH_SIZE = np.uint64(0).nbytes  # 8 bytes


def compute_hashes(batch: Dict[str, Any], idx: List[int], column: str, hash_func: Callable) -> Dict[str, Any]:
    """
    Compute a hash for each line in the document.

    Parameters
    ----------
    batch : Dict[str, Any]
        A batch of one example.
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
    lines = batch[column][0].split("\n")
    n = len(lines)
    hashes = [hash_func(bytes(normalize_for_dedup(line), encoding="utf-8")) for line in lines]
    return {
        "__hash__": hashes,
        "__id__": [idx[0] for _ in range(n)],
        "__idx__": list(range(n)),
    }


def dedup(record: Dict[str, Any], idx: int, column: str, lookup: Dict) -> Dict[str, Any]:
    """
    Remove duplicated lines from the document.

    Parameters
    ----------
    record : Dict[str, Any]
        A record of one example.
    idx : int
        The index of the example in the dataset.
    column : str
        The column name of the text.
    lookup : Dict
        A dictionary containing duplicated (example index, line index) pairs.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the deduplicated record.
    """
    lines = record[column].split("\n")
    new_content = []
    for j, line in enumerate(lines):
        if (idx, j) in lookup:
            continue
        new_content.append(line)
    record[column] = "\n".join(new_content)
    return record


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="text_dedup.ccnet",
        description="Deduplicate line-level text using exact hashing",
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
                num_proc=args.num_proc,
            )

        def md5_digest_sized(data: bytes) -> bytes:
            return md5_digest(data)[:HASH_SIZE]

        def sha256_digest_sized(data: bytes) -> bytes:
            return sha256_digest(data)[:HASH_SIZE]

        def xxh3_digest_sized(data: bytes) -> bytes:
            return xxh3_128_digest(data)[:HASH_SIZE]

        hash_func = {
            "md5": md5_digest,
            "sha256": sha256_digest,
            # xxh3 is much faster when used raw
            "xxh3": xxh3_64_digest if HASH_SIZE == 8 else xxh3_digest_sized,
        }[args.hash_func]

        LEN_DATASET = len(ds)
        hashes = set()
        remove = set()

        with timer("Processing"):
            hashed = ds.map(
                compute_hashes,
                batched=True,
                batch_size=1,
                with_indices=True,
                num_proc=args.num_proc,
                fn_kwargs={"column": args.column, "hash_func": hash_func},
                remove_columns=ds.column_names,
                desc="Computing hashes...",
            )

            NUM_SHARDS = int(np.ceil(len(hashed) / args.batch_size))
            for batch_idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = hashed.shard(NUM_SHARDS, batch_idx, contiguous=True)
                for h, id_, idx in tqdm(
                    zip(ds_shard["__hash__"], ds_shard["__id__"], ds_shard["__idx__"]),
                    leave=False,
                ):
                    if h in hashes:
                        remove.add((id_, idx))
                        continue
                    hashes.add(h)

        with timer("Filtering"):
            # TODO: remove might pose a memory bottleneck
            ds = ds.map(
                dedup,
                with_indices=True,
                num_proc=args.num_proc,
                fn_kwargs={"column": args.column, "lookup": remove},
                desc="Deduping",
            )
            ds = ds.filter(lambda x: len(x[args.column]) > 0, num_proc=args.num_proc, desc="Filtering 0 length docs")

        with timer("Saving"):
            ds.save_to_disk(args.output)

        with timer("Cleaning"):
            if args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before document count':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'Before line count':<{PAD}}: {len(hashed)}")
    logger.info(f"{'After document count':<{PAD}}: {len(ds)}")
