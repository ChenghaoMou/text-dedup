#!/usr/bin/env python
# @Date         : 2023-05-06 19:34:35
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
# @Description  : Line-level deduplication based on Exact Hashing
# @Reference    : https://github.com/facebookresearch/cc_net/blob/main/cc_net/dedup.py

import multiprocessing as mp
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import click
import numpy as np
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import ExactHashArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import Timer
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import md5_digest
from text_dedup.utils import normalize as normalize_for_dedup
from text_dedup.utils import sha256_digest
from text_dedup.utils import xxh3_64_digest
from text_dedup.utils import xxh3_128_digest

HASH_SIZE = np.uint64(0).nbytes  # 8 bytes
mp.set_start_method("fork", force=True)
HASH_COLUMN = "__hash__"
ID_COLUMN = "__id__"


def compute_hashes(
    batch: Dict[str, Any], idx: List[int] | None, column: str, hash_func: Callable, idx_column: str | None = None
) -> Dict[str, Any]:
    """
    Compute a hash for each line in the document.

    Parameters
    ----------
    batch : Dict[str, Any]
        A batch of one example.
    idx : List[int] | None
        The index of the example in the dataset.
    column : str
        The column name of the text.
    hash_func : Callable
        The hash function to use.
    idx_column : str | None
        The column name of the index.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the hashes, the index of the example, and the index of the lines.
    """
    lines = batch[column][0].split("\n")
    idx = idx[0] if idx is not None else batch[idx_column][0]
    n = len(lines)
    hashes = [hash_func(bytes(normalize_for_dedup(line), encoding="utf-8")) for line in lines]
    return {
        HASH_COLUMN: hashes,
        ID_COLUMN: [idx for _ in range(n)],
        INDEX_COLUMN: list(range(n)),
    }


def dedup(
    record: Dict[str, Any], idx: int | None, column: str, lookup: Dict, idx_column: str | None = None
) -> Dict[str, Any]:
    """
    Remove duplicated lines from the document.

    Parameters
    ----------
    record : Dict[str, Any]
        A record of one example.
    idx : int | None
        The index of the example in the dataset.
    column : str
        The column name of the text.
    lookup : Dict
        A dictionary containing duplicated (example index, line index) pairs.
    idx_column : str | None
        The column name of the index.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the deduplicated record.
    """
    lines = record[column].split("\n")
    idx = idx if idx is not None else record[idx_column]
    new_content = []
    for j, line in enumerate(lines):
        if (idx, j) in lookup:
            continue
        new_content.append(line)
    record[column] = "\n".join(new_content)
    return record


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@ExactHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    exact_hash_args: ExactHashArgs,
):
    timer = Timer()

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
    }[exact_hash_args.hash_func]

    with timer("Total"):
        with timer("Loading"):
            ds, _ = load_hf_dataset(io_args=io_args, meta_args=meta_args)

        LEN_DATASET = len(ds)
        hashes = set()
        remove = set()

        with timer("Processing"):
            hashed = ds.map(
                compute_hashes,
                batched=True,
                batch_size=1,
                with_indices=False,
                num_proc=io_args.num_proc,
                fn_kwargs={"column": meta_args.column, "hash_func": hash_func, "idx_column": INDEX_COLUMN, "idx": None},
                remove_columns=[c for c in ds.column_names if c != INDEX_COLUMN],
                desc="Computing hashes...",
            )
            NUM_SHARDS = int(np.ceil(len(hashed) / meta_args.batch_size))
            for batch_idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = hashed.shard(NUM_SHARDS, batch_idx, contiguous=True)
                for h, id_, idx in tqdm(
                    zip(ds_shard[HASH_COLUMN], ds_shard[ID_COLUMN], ds_shard[INDEX_COLUMN]),
                    leave=False,
                ):
                    if h in hashes:
                        remove.add((id_, idx))
                        continue
                    hashes.add(h)

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                dedup,
                with_indices=True,
                num_proc=io_args.num_proc,
                fn_kwargs={"column": meta_args.column, "lookup": remove},
                desc="Deduping",
            )
            ds = ds.filter(
                lambda x: len(x[meta_args.column]) > 0, num_proc=io_args.num_proc, desc="Filtering 0 length docs"
            )

        with timer("Saving"):
            ds.save_to_disk(io_args.output)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before document count':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'Before line count':<{PAD}}: {len(hashed)}")
    logger.info(f"{'After document count':<{PAD}}: {len(ds)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
