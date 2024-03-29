#!/usr/bin/env python
# @Date    : 2022-11-05 11:03:18
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
from collections import defaultdict
from itertools import permutations
from typing import Any
from typing import Callable

import click
import datasets
import numpy as np
from bitarray import bitarray
from bitarray import frozenbitarray
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import CLUSTER_COLUMN
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import ngrams
from text_dedup.utils import xxh3_64_digest
from text_dedup.utils import xxh3_128_digest

mp.set_start_method("fork", force=True)
datasets.logging.set_verbosity_error()
uf = UnionFind()
KEY_COLUMN = "__keys__"
SIGNATURE_COLUMN = "__signature__"


def _hamming_distance(a: bitarray, b: bitarray) -> int:
    """
    Compute the Hamming distance between two bitarrays.

    Parameters
    ----------
    a : bitarray
        The first bitarray.
    b : bitarray
        The second bitarray.

    Returns
    -------
    int
        The Hamming distance between the two bitarrays.

    Examples
    --------
    >>> _hamming_distance(bitarray("1010"), bitarray("1010"))
    0
    >>> _hamming_distance(bitarray("1010"), bitarray("0010"))
    1
    """
    return (a ^ b).count(1)


class Permutation:
    def __init__(self, f: int, k: int, b: int, masks: list[tuple[bitarray, int, int, int]]) -> None:
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
        self.widths: list[int] = []  # block widths
        self.offsets: list[int] = []  # block offsets
        self.reverse_masks: list[int] = []  # block reverse masks
        self.masks: list[bitarray] = []  # block masks
        for mask, mask_size, start, _ in masks:
            self.widths.append(mask_size)
            offset = start - width
            width += mask_size
            self.offsets.append(offset)
            if offset > 0:
                self.reverse_masks.append(mask << offset)
            else:
                self.reverse_masks.append(mask >> -offset)

            self.masks.append(mask)

        assert (
            sum(self.widths) == f
        ), f"The sum of block widths {sum(self.widths)} must be equal to the fingerprint size {f}"

        prefix_width = sum(self.widths[: b - k])
        self.search_mask: bitarray = bitarray(f)
        self.search_mask.setall(0)
        self.search_mask[:prefix_width] = 1
        self.search_mask = frozenbitarray(self.search_mask)

    def permute(self, x: bitarray) -> bitarray:
        """
        Permute the fingerprint.

        Parameters
        ----------
        x: bitarray
            The fingerprint to be permuted

        Returns
        -------
        bitarray
            The permuted fingerprint
        """
        result = bitarray(self.f)
        result.setall(0)

        for mask, offset in zip(self.masks, self.offsets):
            assert len(mask) == len(x) == self.f, f"{len(mask)}, {len(x)}, {self.f}"
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
        result = bitarray(self.f)
        result.setall(0)
        for mask, offset in zip(self.reverse_masks, self.offsets):
            if offset > 0:
                result |= (x & mask) >> offset
            else:
                result |= (x & mask) << -offset
        return result


def _create_permutations(f: int, k: int, b: int) -> list[Permutation]:
    """
    Create permutations for f bits and b blocks with k-bit difference allowed.

    Parameters
    ----------
    f: int
        Fingerprint size
    k: int
        Bit difference allowed
    b: int
        Number of blocks

    Returns
    -------
    List[Permutation]
        The permutations

    Examples
    --------
    >>> from bitarray.util import urandom
    >>> perms = _create_permutations(128, 3, 4)
    >>> len(perms)
    4
    >>> data = urandom(128)
    >>> for perm in perms:
    ...     assert perm.reverse(perm.permute(data)) == data, f"{perm.reverse(perm.permute(data))} != {data}"
    """
    masks: list[tuple[bitarray, int, int, int]] = []

    max_block_size: int = math.ceil(f / b)
    min_block_size: int = math.floor(f / b)

    # solve: max_block_size * x + min_block_size * y == f
    x, y = 0, 0
    while True:
        x += 1
        if (f - x * max_block_size) % min_block_size == 0:
            y = (f - x * max_block_size) // min_block_size
            break

    assert (
        x * max_block_size + y * min_block_size == f
    ), f"{x=} w/ {max_block_size}, {y=} w/ {min_block_size} are invalid"

    start = end = 0

    for i in range(b):
        block_size = max_block_size if x > 0 else min_block_size
        start, end = end, min(end + block_size, f)
        if start >= end:
            break
        mask: bitarray = bitarray(f)
        mask.setall(0)
        mask[start:end] = 1
        masks.append(
            (
                mask,
                end - start,
                start,
                end,
            )
        )

    results: list[Permutation] = []
    # b - k many blocks must be the same
    indices = set(range(len(masks)))
    for leading_idx in permutations(range(len(masks)), b - k):
        remaining_idx = sorted(indices - set(leading_idx))
        blocks = [masks[i] for i in leading_idx] + [masks[i] for i in remaining_idx]
        results.append(Permutation(f, k, b, blocks))

    return results


def _unsigned_hash(obj: bytes, hash_func: Callable) -> bitarray:
    """
    Compute a hash of an object.

    It doesn't really matter what hash function to use, as long as it is consistent.

    Parameters
    ----------
    obj: bytes
        The object to hash.
    hash_func: Callable
        The hash function to use.

    Returns
    -------
    bitarray
        The hash of the object.

    Examples
    --------
    >>> len(_unsigned_hash(b"hello world", xxh3_64_digest))
    64
    >>> len(_unsigned_hash(b"hello world", xxh3_128_digest))
    128
    """
    result = bitarray(0)
    result.frombytes(hash_func(obj))
    return result


def compute(hashes: list[bitarray]) -> bitarray:
    """
    Compute the Simhash of a list of hashes.

    Notes to myself: You tried porting this to Cython, but it didn't improve the performance.
    Others have experimented with numpy types and operators, but it didn't improve performance

    Parameters
    ----------
    hashes : List[int]
        The list of hashes.

    Returns
    -------
    bitarray
        The Simhash of the list of hashes.

    Examples
    --------
    >>> from bitarray.util import int2ba, ba2int
    >>> res = compute([int2ba(13352372148217134600, length=64), int2ba(5020219685658847592, length=64)])
    >>> ba2int(res)
    74633958390507528
    """
    sigs = np.asarray([h.tolist() for h in hashes], dtype=int)
    sig = np.where(np.sum(2 * sigs - 1, axis=0) > 0, True, False)
    res = bitarray()
    res.pack(sig.tobytes())
    return res


def embed_func(
    content: str,
    idx: int,
    *,
    ngram: int,
    permutations: list[Permutation],
    hash_func: Callable,
) -> dict[str, Any]:
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
    hash_func : Callable
        hash function to use

    Returns
    -------
    Dict[str, Any]
        The simhash signature and the index of the text as a dictionary.

    Examples
    --------
    >>> res = embed_func("hello world", 0, ngram=3, permutations=None, hash_func=xxh3_64_digest)
    >>> res[INDEX_COLUMN]
    0
    >>> len(res[SIGNATURE_COLUMN])
    8
    """
    tokens = {bytes("".join(ng).lower(), "utf-8") for ng in ngrams(list(content), n=ngram)}
    sig = compute([_unsigned_hash(t, hash_func) for t in tokens])
    keys: list[tuple[bytes, bytes]] = []
    if permutations:
        for permutation in permutations:
            keys.append(
                (
                    permutation.search_mask.tobytes(),
                    (permutation.permute(sig) & permutation.search_mask).tobytes(),
                )
            )
    return {INDEX_COLUMN: idx, KEY_COLUMN: keys, SIGNATURE_COLUMN: sig.tobytes()}


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@SimHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    simhash_args: SimHashArgs,
):
    global uf
    uf.reset()
    timer = Timer()
    PERMUTATIONS = _create_permutations(simhash_args.f, k=simhash_args.bit_diff, b=simhash_args.num_bucket)
    BUCKETS: dict[Any, list] = defaultdict(list)

    # current code only supports 64 or 128 bit simhash sizes
    hash_func = {64: xxh3_64_digest, 128: xxh3_128_digest}.get(simhash_args.f, xxh3_64_digest)

    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)

        LEN_DATASET = len(ds)  # type: ignore

        with timer("SimHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "ngram": simhash_args.ngram,
                    "permutations": PERMUTATIONS,
                    "hash_func": hash_func,
                },
                input_columns=[meta_args.column, INDEX_COLUMN],
                remove_columns=[meta_args.column],
                num_proc=io_args.num_proc,  # type: ignore
                with_indices=False,
                desc="SimHashing...",  # type: ignore
            )

        # TODO Create multiple BUCKETS for parallelization
        LEN_EMBEDDED = len(embedded)
        NUM_SHARDS = np.ceil(LEN_EMBEDDED / meta_args.batch_size).astype(int)
        with timer("Clustering"):
            for batch_idx in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating SimHashes...",
            ):
                # Iterate over each batch dataset from the total hash embedded dataset
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS, index=batch_idx, contiguous=True, writer_batch_size=meta_args.batch_size
                )
                for idx, keys, sig in tqdm(
                    zip(embedded_shard[INDEX_COLUMN], embedded_shard[KEY_COLUMN], embedded_shard[SIGNATURE_COLUMN]),
                    desc="Indexing...",
                    leave=False,
                    total=len(embedded_shard),
                ):
                    # each example in a batch has id, key, signatures we built above
                    # we extract that signature and make it a frozen bitarray
                    sig = frozenbitarray(buffer=sig)
                    # we make a neighbor set for each batch
                    neighbors = set()
                    # iterate for for each permutation key
                    for key in keys:
                        key = tuple(key)
                        # BUCKETS is a defaultdict defaulting to empty list
                        # this iterates over indicies ALREADY in the bucket
                        for idy, other_fingerprint in BUCKETS[key]:
                            if idy in neighbors:
                                continue
                            if _hamming_distance(sig, other_fingerprint) <= simhash_args.bit_diff:
                                uf.union(idx, idy)
                                neighbors.add(idy)
                        BUCKETS[key].append((idx, sig))

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=io_args.num_proc,  # type: ignore
                new_fingerprint=str(random.getrandbits(128)),  # type: ignore
                desc="Finding clusters...",  # type: ignore
            )
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
                num_proc=io_args.num_proc,
                desc="Filtering clusters...",
            )

        with timer("Saving"):
            final_data = final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])
            final_data.save_to_disk(io_args.output)
            if io_args.debug:
                uf.dump(path=os.path.join(io_args.output, "uf.pkl"), id2id=id2id)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
