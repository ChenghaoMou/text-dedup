#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 11:03:18
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import argparse
import gc
import math
import multiprocessing
import os
import random
import warnings
from collections import defaultdict
from itertools import permutations
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Set
from typing import Tuple

import datasets
import numpy as np
import xxhash
from datasets import load_dataset
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import UnionFind
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils import add_simhash_args
from text_dedup.utils import ngrams
from text_dedup.utils.timer import Timer

warnings.filterwarnings("ignore", category=FutureWarning)
multiprocessing.set_start_method("fork", force=True)
datasets.logging.set_verbosity_error()

# With multiprocessing and copy-on-write fork (Linux and macOS),
# we can use global variables to share objects across processes.
# This might not be the case on some systems where objects are
# pickled and sent to the child processes. It might also not be reflected
# when you use top command to check the memory usage. One way to check is to
# print the id of the object in the child processes and see if they are the same.
# References:
# 1. https://stackoverflow.com/questions/38084401/leveraging-copy-on-write-to-copy-data-to-multiprocessing-pool-worker-process
# 2. https://stackoverflow.com/questions/53841599/python-multiprocessing-copy-on-write-behaving-differently-between-osx-and-ubuntu
# 3. https://stackoverflow.com/questions/40221868/multiprocessing-global-variable-memory-copying
# 4. https://docs.python.org/3/library/gc.html#gc.freeze


simhash_index: SimHashIndex | None = None
dup_ids: Set[int] | None = None

BIT_MASK: np.ndarray = 2 ** np.arange(64, dtype=np.uint64).reshape([1, 64])


def _hamming_distance(a: int, b: int) -> int:
    """
    Compute the Hamming distance between two integers.

    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.

    Returns
    -------
    int
        The Hamming distance between the two integers.

    Examples
    --------
    >>> _hamming_distance(0b1010, 0b0101)
    4
    >>> _hamming_distance(0b1010, 0b1010)
    0
    """
    return (a ^ b).bit_count()


class Permutation:
    def __init__(self, f: int, k: int, b: int, masks: List[Tuple[int, int, int, int]]) -> None:
        """
        A permutation object for bit manipulation.

        More details about this permutation can be found in https://github.com/seomoz/simhash-py#architecture.

        Parameters
        ----------
        f: int
            The fingerprint bit length
        k: int
            The bit difference allowed
        b: int
            The number of blocks
        masks:
            The block masks generated from `_create_permutations`
        """
        self.f = f
        self.k = k
        self.b = b

        width: int = 0
        self.widths: List[int] = []  # block widths
        self.offsets: List[int] = []  # block offsets
        self.reverse_masks: List[int] = []  # block reverse masks
        self.masks: List[int] = []  # block masks
        for mask, mask_size, start, _ in masks:
            self.widths.append(mask_size)
            width += mask_size
            offset = f - width - start
            self.offsets.append(offset)
            if offset > 0:
                self.reverse_masks.append(mask << offset)
            else:
                self.reverse_masks.append(mask >> -offset)

            self.masks.append(mask)

        prefix_width = sum(self.widths[: b - k])
        self.search_mask: int = 0
        for i in range(f):
            if i < prefix_width:
                self.search_mask += 1
            self.search_mask <<= 1

    def permute(self, x: int) -> int:
        """
        Permute the fingerprint.

        Parameters
        ----------
        x: int
            The fingerprint to be permuted

        Returns
        -------
        int
            The permuted fingerprint
        """
        result = 0

        for mask, offset in zip(self.masks, self.offsets):
            if offset > 0:
                result |= (x & mask) << offset
            else:
                result |= (x & mask) >> -offset

        return result

    def reverse(self, x: int) -> int:
        """
        Reverse the permutation.

        Parameters
        ----------
        x: int
           The fingerprint to be reversed

        Returns
        -------
        int
            The reversed fingerprint
        """
        result = 0
        for mask, offset in zip(self.reverse_masks, self.offsets):
            if offset > 0:
                result |= (x & mask) >> offset
            else:
                result |= (x & mask) << -offset
        return result


def _create_permutations(f: int, k: int, b: int) -> List[Permutation]:
    """
    Create permutations for f bit, b blocks, and k bit difference allowed.

    Parameters
    ----------
    f: int
        The fingerprint to be permuted
    k: int
        The bit difference allowed
    b: int
        The number of blocks

    Returns
    -------
    List[Permutation]
        The permutations
    """
    block_size: int = math.ceil(f / b)
    masks: List[Tuple[int, int, int, int]] = []
    for i in range(b):
        mask: int = 0
        start, end = i * block_size, min((i + 1) * block_size, f)
        for j in range(start, end):
            mask |= 1 << j
        masks.append(
            (
                mask,
                end - start,
                start,
                end,
            )
        )

    results: List[Permutation] = []
    for leading_blocks in permutations(masks, b - k):
        blocks = list(leading_blocks)
        for record in masks:
            if record not in blocks:
                blocks.append(record)
        results.append(Permutation(f, k, b, blocks))

    return results


class SimHashIndex(object):
    def __init__(
        self,
        f: int = 64,
        k: int = 3,
        b: int = 4,
    ):
        assert b > k, "b must be greater than k"

        self.k = k
        self.b = b
        self.f = f

        self.bucket: Dict[Any, List] = defaultdict(list)
        self.permutations = _create_permutations(f, k, b)

    def query(self, fingerprint: int) -> List[Any]:
        fingerprint = int(fingerprint)
        ans = set()
        for key in self.get_keys(fingerprint):
            for idx, other_fingerprint in self.bucket[key]:
                if _hamming_distance(fingerprint, other_fingerprint) <= self.k:
                    ans.add(idx)
        return list(ans)

    def add(self, idx: int, fingerprint: int):
        fingerprint = int(fingerprint)
        for key in self.get_keys(fingerprint):
            self.bucket[key].append((idx, fingerprint))

    def get_keys(self, fingerprint: int) -> Generator[Tuple[int, int], None, None]:
        fingerprint = int(fingerprint)
        for permutation in self.permutations:
            yield permutation.search_mask, permutation.permute(fingerprint) & permutation.search_mask


def unpackbits(x: np.ndarray, num_bits: int = 64) -> np.ndarray:
    """
    Unpack a numpy integer array into a numpy bit array.

    Parameters
    ----------
    x: np.ndarray
        The numpy integer array, unsigned
    num_bits: int
        The number of bits to unpack

    Returns
    -------
    np.ndarray
        The numpy bit array

    Examples
    --------
    >>> unpackbits(np.array([0, 1], dtype=np.uint64)).shape
    (2, 64)
    """
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    return (x & BIT_MASK).astype(bool).astype(int).reshape(xshape + [num_bits])


def _unsigned_hash(obj: bytes) -> int:
    """
    Compute a 64-bit hash of an object.

    It doesn't really matter what hash function to use, as long as it is consistent.

    Parameters
    ----------
    obj: bytes
        The object to hash.

    Returns
    -------
    int
        The hash of the object.

    Examples
    --------
    >>> _unsigned_hash(b'hello world')
    5020219685658847592
    """
    return xxhash.xxh64(obj).intdigest()


def compute(hashes: List[int]) -> int:
    """
    Compute the Simhash of a list of hashes.

    Notes to myself: You tried porting this to Cython, but it didn't improve the performance.

    Parameters
    ----------
    hashes : List[int]
        The list of hashes.

    Returns
    -------
    int
        The Simhash of the list of hashes.

    Examples
    --------
    >>> compute([13352372148217134600, 5020219685658847592])
    74633958390507528
    """
    bits = 2 * unpackbits(np.asarray(hashes, dtype=np.uint64), 64) - 1
    res = (np.where(np.sum(bits, axis=0) > 0, 1, 0)[::-1]).astype(np.uint64)
    return np.packbits(res).view(np.uint64).byteswap().item()


def embed_func(content: str, idx: int, *, ngram: int) -> Dict[str, Any]:
    """
    Calculate the simhash signature of a text.

    Parameters
    ----------
    content : str
        The text to be hashed.
    idx : int
        The index of the text.
    ngram : int
        The ngram size.

    Returns
    -------
    Dict[str, Any]
        The simhash signature and the index of the text as a dictionary.

    Examples
    --------
    >>> res = embed_func("hello world", 0, ngram=3)
    >>> res["__id__"]
    0
    >>> res["__signature__"].dtype
    dtype('uint64')
    """
    tokens = ngrams(content.split(" "), n=ngram)
    sig = compute([_unsigned_hash(" ".join(t).encode("utf-8")) for t in tokens])
    return {"__signature__": np.uint64(sig), "__id__": idx}


def query_func(idx: int, signature: np.uint64, *, index: SimHashIndex) -> Dict[str, Any]:
    """
    Query the simhash index.

    Parameters
    ----------
    idx : int
        The index of the text.
    signature : np.ndarray
        The simhash signature of the text.
    index : MinHashLSH
        The simhash index.
    seed : int
        The seed for the simhash.

    Returns
    -------
    Dict[str, Any]
        The neighbors of the text as a dictionary.

    Examples
    --------
    >>> index = SimHashIndex(f=64, k=3, b=4)
    >>> h = embed_func("hello world", 0, ngram=3)["__signature__"]
    >>> h.dtype
    dtype('uint64')
    >>> index.add(0, h)
    >>> index.add(1, h)
    >>> res = query_func(0, h, index=index)
    >>> res["__id__"]
    0
    >>> res["__neighbors__"]
    [1]
    """
    return {
        "__neighbors__": [dup_idx for dup_idx in index.query(signature) if dup_idx != idx],
        "__id__": idx,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="text_dedup.simhash",
        description="Deduplicate text using simhash",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_simhash_args(parser)

    args = parser.parse_args()
    simhash_index = SimHashIndex(
        k=args.bit_diff,
        b=args.num_bucket,
    )
    timer = Timer()
    uf = UnionFind()

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
                use_auth_token=args.use_auth_token,
            )

        DATA_SIZE = len(ds)

        with timer("SimHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={"ngram": args.ngram},
                input_columns=[args.column],
                remove_columns=[args.column],
                num_proc=os.cpu_count(),
                with_indices=True,
                desc=f"SimHashing...",
            )

        with timer("Indexing"):
            for i in tqdm(
                range(0, len(embedded), args.batch_size),
                dynamic_ncols=True,
                desc="Iterating MinHashes...",  # noqa: E501
            ):
                batch = embedded[i : i + args.batch_size]
                for idx, sig in tqdm(
                    zip(batch["__id__"], batch["__signature__"]),
                    desc="Indexing signatures...",
                    leave=False,
                    total=len(batch["__id__"]),
                ):
                    for neighbor in simhash_index.query(sig):
                        uf.union(idx, neighbor)
                    simhash_index.add(idx, sig)

        with timer("Filtering"):
            gc.freeze()
            gc.disable()
            ds = ds.map(
                function=lambda _, idx: {"__cluster__": uf.find(idx)},
                with_indices=True,
                num_proc=os.cpu_count(),
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            gc.enable()
            gc.collect()
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            final_data = ds.filter(
                function=lambda record, idx: record["__cluster__"] == idx,
                with_indices=True,
                num_proc=os.cpu_count(),
                desc="Filtering clusters...",
            )

        with timer("Saving"):
            final_data = final_data.remove_columns(["__cluster__"])
            final_data.save_to_disk(args.output)

    FINAL_DATA_SIZE = len(final_data)
    DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
    PAD = 32

    for key, value in timer.elapsed_times.items():
        logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
    logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
    logger.info(f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})")  # noqa: E501
    logger.info(f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})")  # noqa: E501
    logger.info("🤗 Happy Deduplicating 🤗")
