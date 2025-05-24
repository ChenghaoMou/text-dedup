#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
from typing import Any
from typing import Callable

import click
import datasets
import numpy as np
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import CLUSTER_COLUMN
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import ngrams
from text_dedup.utils import optimal_param
from text_dedup.utils import sha1_hash
from text_dedup.utils import xxh3_16hash
from text_dedup.utils import xxh3_32hash

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()
# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)
uf = UnionFind()
SIGNATURE_COLUMN = "__signatures__"


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    hash_func: Callable,
    dtype: type,
    max_hash: np.uint,
    modulo_prime: np.uint,
) -> dict[str, Any]:
    """
    Calculate hash values for the content.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    min_length : int
        The minimum length of the document in terms of tokens.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    hash_func : Callable
        The hash function to use.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> max_hash = np.uint32((1 << 32) - 1)
    >>> modulo_prime = np.uint32((1 << 32) - 5)
    >>> PERMUTATIONS = (RNG.randint(1, modulo_prime, size=num_perm), RNG.randint(0, modulo_prime, size=num_perm))
    >>> res = embed_func(
    ...     content,
    ...     idx,
    ...     num_perm=num_perm,
    ...     ngram_size=ngram_size,
    ...     min_length=0,
    ...     hashranges=hashranges,
    ...     permutations=PERMUTATIONS,
    ...     hash_func=xxh3_32hash,
    ...     dtype=np.uint32,
    ...     max_hash=max_hash,
    ...     modulo_prime=modulo_prime,
    ... )
    >>> len(res[SIGNATURE_COLUMN])
    10
    >>> res[INDEX_COLUMN]
    0
    """
    # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
    # the formula is a * x(base hash of each shingle) + b
    a, b = permutations
    # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
    # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
    tokens: set[bytes] = {
        bytes(" ".join(t).lower(), "utf-8") for t in ngrams(NON_ALPHA.split(content.lower()), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array([hash_func(token) for token in tokens], dtype=dtype).reshape(len(tokens), 1)
    # Permute the hash values to produce new universal hashes
    # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
    # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    # this part is where the name "min" of minhash comes from
    # this stacks all the hashes and then takes the minimum from each column
    masks: np.ndarray = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    # Originally, byteswap was done for speed. Testing show it has a negligible impact
    # keeping  for backward compatibility, even though theoretically and empirically
    # it doesnt matter if it is there or not. github.com/ekzhu/datasketch/issues/114
    Hs: list[bytes] = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {SIGNATURE_COLUMN: Hs, INDEX_COLUMN: idx}


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@MinHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    minhash_args: MinHashArgs,
):
    global uf
    uf.reset()
    HASH_BITS: int = minhash_args.hash_bits

    # 64 bit config is backwards compatibility mode.
    # it uses 64 bit types but almost entirely 32bit data, except for one mersenne prime 2^61
    # why legacy implementations used mersenne primes for modulo:
    # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16 bit config does not use a mersenne prime.
        # The original reason for using mersenne prime was speed.
        # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    # defaults to backwards compatible HASH_BITS = 64, which is np.uint64 dtypes with 32bit hashes
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(HASH_BITS, HASH_CONFIG[64])

    match minhash_args.hash_func:
        case "sha1":

            def hash_func(byte_data):
                return sha1_hash(byte_data, d=min(HASH_BITS, 32))

        case "xxh3":
            if HASH_BITS == 16:
                hash_func = xxh3_16hash
            else:
                hash_func = xxh3_32hash

    timer = Timer()

    if minhash_args.b is not None and minhash_args.r is not None:
        B, R = minhash_args.b, minhash_args.r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
        # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
        # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
        # Both effects move the result towards more documents being considered duplicates.
        B, R = optimal_param(
            minhash_args.threshold,
            minhash_args.num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES: list[dict[int, set]] = [defaultdict(set) for _ in range(B)]

    # for minhash, we need to make a lot of hashes(=num_perms).
    # In many previous implementations, this is achieved through a method described in
    # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
    # There we start with a know good hash x (=hash_func) and permutate it as the following:
    # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
    # the following produces these a, b pairs
    PERMUTATIONS: tuple[np.ndarray, np.ndarray] = (
        RNG.randint(
            1, MODULO_PRIME, size=(minhash_args.num_perm,), dtype=DTYPE
        ),  # a is a multiplier so should not be 0
        RNG.randint(0, MODULO_PRIME, size=(minhash_args.num_perm,), dtype=DTYPE),  # b
    )

    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[meta_args.column].lower())) >= minhash_args.min_length,
                num_proc=io_args.num_proc,
            )

        LEN_DATASET = len(ds)

        with timer("MinHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": minhash_args.num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": minhash_args.ngram,
                    "min_length": minhash_args.min_length,
                    "permutations": PERMUTATIONS,
                    "hash_func": hash_func,
                    "dtype": DTYPE,
                    "max_hash": MAX_HASH,
                    "modulo_prime": MODULO_PRIME,
                },
                input_columns=[meta_args.column, INDEX_COLUMN],
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                num_proc=io_args.num_proc,
                with_indices=False,
                desc="Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            NUM_SHARDS = np.ceil(LEN_EMBEDDED / meta_args.batch_size).astype(int)

        with timer("Clustering"):
            edges = []
            for i in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating MinHashes...",  # noqa: E501
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=meta_args.batch_size,
                )
                for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            logger.info(f"Number of clusters: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                # cluster: Set[int]
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)
            logger.info(f"Number of edges: {len(set(edges))}")

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=io_args.num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
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
                uf.dump(os.path.join(io_args.output, "uf.pkl"), id2id=id2id)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
