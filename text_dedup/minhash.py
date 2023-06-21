#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import argparse
import gc
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import datasets
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from scipy.integrate import quad as integrate
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import UnionFind
from text_dedup.utils import ngrams
from text_dedup.utils.add_args import add_io_args
from text_dedup.utils.add_args import add_meta_args
from text_dedup.utils.add_args import add_minhash_args
from text_dedup.utils.timer import Timer

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
datasets.logging.set_verbosity_error()


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    >>> sha1_hash(b"hello world", 128)
    310522945683037930239412421226792791594
    """
    return int.from_bytes(hashlib.sha1(data).digest()[: d // 8], byteorder="little")


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
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
    >>> PERMUTATIONS = np.array(
    ...     [
    ...         (
    ...             RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
    ...             RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
    ...         )
    ...         for _ in range(num_perm)
    ...     ],
    ...     dtype=np.uint64,
    ... ).T
    >>> res = embed_func(content, idx, num_perm=num_perm, ngram_size=ngram_size, min_length=0, hashranges=hashranges, permutations=PERMUTATIONS)
    >>> len(res["__signatures__"])
    10
    >>> res["__id__"]
    0
    """
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size, min_length)}
    hashvalues: np.ndarray = np.array([sha1_hash(token.lower().encode("utf-8")) for token in tokens], dtype=np.uint64)
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.

    Examples
    --------
    >>> optimal_param(0.75, 256)
    (21, 12)
    >>> optimal_param(0.75, 256, 0.1, 0.9)
    (28, 9)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(
        prog="text_dedup.minhash",
        description="Deduplicate text using minhash",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_minhash_args(parser)
    args = parser.parse_args()

    mp.set_start_method("fork", force=True)
    uf = UnionFind()
    timer = Timer()

    if args.b is not None and args.r is not None:
        B, R = args.b, args.r
    else:
        B, R = optimal_param(args.threshold, args.num_perm)

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES: List[Dict[int, Set]] = [defaultdict(set) for _ in range(B)]

    with timer("Total"):
        with timer("Loading"):
            if args.local:
                ds = load_from_disk(args.path)
            else:
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
        PERMUTATIONS = np.array(
            [
                (
                    RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(args.num_perm)
            ],
            dtype=np.uint64,
        ).T

        with timer("MinHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": args.num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": args.ngram,
                    "min_length": args.min_length,
                    "permutations": PERMUTATIONS,
                },
                input_columns=[args.column],
                remove_columns=ds.column_names,
                num_proc=os.cpu_count(),
                with_indices=True,
                desc="Fingerprinting...",
            )

        with timer("Clustering"):
            for i in tqdm(
                range(0, len(embedded), args.batch_size),
                dynamic_ncols=True,
                desc="Iterating MinHashes...",  # noqa: E501
            ):
                batch = embedded[i : i + args.batch_size]
                for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        uf.union(x, idx)

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
            if args.debug:
                with open(os.path.join(args.output, "uf.pkl"), "wb") as f:
                    pickle.dump(uf, f, protocol=pickle.HIGHEST_PROTOCOL)

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {len(ds)}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")
