#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 11:03:18
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Set

import datasets
import dill as pickle
import networkit as nk
import numpy as np
from datasets import load_dataset
from datasketch import LeanMinHash, MinHash, MinHashLSH
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import (
    add_io_args,
    add_meta_args,
    add_minhash_args,
    find_duplicate_components,
    ngrams,
)

warnings.filterwarnings("ignore", category=FutureWarning)
multiprocessing.set_start_method("fork", force=True)
datasets.logging.set_verbosity_error()
nk.setLogLevel("ERROR")

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

lsh: MinHashLSH | None = None
dup_ids: Set[int] | None = None


def embed_func(content: str, idx: int, *, num_perm: int, seed: int, ngram: int) -> Dict[str, Any]:
    """
    Calculate the minhash signature of a text.

    Parameters
    ----------
    content : str
        The text to be hashed.
    idx : int
        The index of the text.
    num_perm : int
        The number of permutations.
    seed : int
        The seed for the minhash.
    ngram : int
        The ngram size.

    Returns
    -------
    Dict[str, Any]
        The minhash signature and the index of the text as a dictionary.

    Examples
    --------
    >>> res = embed_func("hello world", 0, num_perm=128, seed=0, ngram=3)
    >>> res["__id__"]
    0
    >>> res["__signature__"].shape
    (128,)
    >>> res["__signature__"].dtype
    dtype('uint64')
    """
    m = MinHash(num_perm=num_perm, seed=seed)
    tokens = ngrams(content, ngram)
    m.update_batch([token.encode("utf-8") for token in tokens])
    return {"__signature__": m.hashvalues, "__id__": idx}


def query_func(idx: int, signature: np.ndarray, *, index: MinHashLSH, seed: int) -> Dict[str, Any]:
    """
    Query the minhash index.

    Parameters
    ----------
    idx : int
        The index of the text.
    signature : np.ndarray
        The minhash signature of the text.
    index : MinHashLSH
        The minhash index.
    seed : int
        The seed for the minhash.

    Returns
    -------
    Dict[str, Any]
        The neighbors of the text as a dictionary.

    Examples
    --------
    >>> lsh = MinHashLSH(threshold=0.5, num_perm=128)
    >>> h = embed_func("hello world", 1, num_perm=128, ngram=3, seed=0)["__signature__"]
    >>> lsh.insert(0, LeanMinHash(hashvalues=h, seed=0))
    >>> lsh.insert(1, LeanMinHash(hashvalues=h, seed=0))
    >>> res = query_func(0, h, index=lsh, seed=0)
    >>> res["__id__"]
    0
    >>> res["__neighbors__"]
    [1]
    """
    return {
        "__neighbors__": [
            dup_idx
            for dup_idx in index.query(
                LeanMinHash(seed=seed, hashvalues=signature),
            )
            if dup_idx != idx  # exclude itself
        ],
        "__id__": idx,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="text_dedup.minhash",
        description="Deduplicate text using minhash",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_minhash_args(parser)

    args = parser.parse_args()

    lsh = MinHashLSH(
        threshold=args.threshold,
        num_perm=args.num_perm,
    )

    assert args.path is not None, "Please specify `path` for `load_dataset`."
    assert args.graph_name is not None, "Please specify `graph_name`."
    assert args.index_name is not None, "Please specify `output_graph`."

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    output_graph = OUTPUT_DIR / args.graph_name
    output_index = OUTPUT_DIR / args.index_name
    output = OUTPUT_DIR / args.dedup_name

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

    DATA_SIZE = len(ds)

    # region: minhash
    elapsed_time["Minhash"] = time.time()
    embedded = ds.map(
        function=embed_func,
        fn_kwargs={"num_perm": args.num_perm, "seed": args.seed, "ngram": args.ngram},
        input_columns=[args.column],
        remove_columns=[args.column],
        num_proc=os.cpu_count(),
        with_indices=True,
        desc=f"MinHashing...",
    )
    elapsed_time["Minhash"] = time.time() - elapsed_time["Minhash"]
    # endregion

    # region: index
    if os.path.exists(output_index) and args.reuse_index:
        elapsed_time["Load Index"] = time.time()
        with open(output_index, "rb") as f:
            lsh = pickle.load(f)
        elapsed_time["Load Index"] = time.time() - elapsed_time["Load Index"]
    else:
        elapsed_time["Index"] = time.time()
        with lsh.insertion_session() as session:
            for data in tqdm(embedded, desc="Indexing signatures..."):
                if data["__id__"] in lsh:
                    continue
                session.insert(
                    data["__id__"],
                    LeanMinHash(seed=args.seed, hashvalues=data["__signature__"]),
                    check_duplication=False,
                )
        elapsed_time["Index"] = time.time() - elapsed_time["Index"]
        elapsed_time["Save Index"] = time.time()
        pickle.dump(lsh, open(output_index, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        elapsed_time["Save Index"] = time.time() - elapsed_time["Save Index"]
    # endregion

    gc.disable()
    gc.freeze()

    # region: query
    elapsed_time["Query"] = time.time()
    queried = embedded.map(
        lambda x, y: query_func(x, y, index=lsh, seed=args.seed),
        num_proc=os.cpu_count(),
        new_fingerprint=str(random.getrandbits(64)),
        input_columns=["__id__", "__signature__"],
        remove_columns=["__signature__"],
        desc=f"Querying...",
    )
    elapsed_time["Query"] = time.time() - elapsed_time["Query"]
    # endregion

    gc.enable()
    gc.unfreeze()
    gc.collect()

    # region: clustering
    elapsed_time["Clustering"] = time.time()
    queried = queried.filter(
        lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
    )
    dup_ids = find_duplicate_components(
        records=queried,
        input_graph=output_graph if args.reuse_graph else None,
        output_graph=output_graph,
    )
    elapsed_time["Clustering"] = time.time() - elapsed_time["Clustering"]
    # endregion

    # region: deduplicate
    elapsed_time["Deduplicate"] = time.time()
    final_data = ds.filter(
        lambda _, idx: idx not in dup_ids,
        num_proc=os.cpu_count(),
        with_indices=True,
        desc="Filtering duplicates...",
    )
    elapsed_time["Deduplicate"] = time.time() - elapsed_time["Deduplicate"]

    elapsed_time["Save"] = time.time()
    final_data.save_to_disk(output)
    elapsed_time["Save"] = time.time() - elapsed_time["Save"]
    # endregion

    elapsed_time["All"] = time.time() - elapsed_time["All"]
    for k, v in elapsed_time.items():
        logger.info(f"{k:<30}: {v:.2f}s")

    logger.info(f"{'Before':<30}: {DATA_SIZE}")
    logger.info(f"{'After':<30}: {len(final_data)}")
    logger.info(f"{'Index':<30}: {output_index}")
    logger.info(f"{'Graph':<30}: {output_graph}")
    logger.info(f"{'Output':<30}: {output}")
