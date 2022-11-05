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
from typing import Any, Dict, Iterable, Set

import datasets
import dill as pickle
import networkit as nk
import numpy as np
from datasets import Dataset, load_dataset
from datasketch import LeanMinHash, MinHash, MinHashLSH
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import add_io_args, add_meta_args, add_minhash_args

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
    m = MinHash(num_perm=num_perm, seed=seed)
    tokens = [t for t in content.split(" ") if t]
    ngrams = [" ".join(tokens[i : i + ngram]) for i in range(0, len(tokens) - ngram + 1, ngram)]
    m.update_batch([token.encode("utf-8") for token in ngrams])
    return {"__signature__": m.hashvalues, "__id__": idx}


def query_func(idx: int, signature: np.ndarray, *, index: MinHashLSH, seed: int) -> Dict[str, Any]:
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


def find_duplicate_components(
    records: Iterable | Dataset,
    input_graph: str | None = None,
    output_graph: str | None = None,
) -> Set[int]:
    if input_graph is not None:
        g = nk.readGraph(str(input_graph), nk.Format.NetworkitBinary)
    else:
        g = nk.graph.Graph()
        for record in tqdm(records, desc="Constructing graph..."):
            for y in record["__neighbors__"]:
                g.addEdge(record["__id__"], y, addMissing=True)

        if output_graph is not None:
            if os.path.exists(output_graph):
                os.remove(output_graph)
            nk.writeGraph(g, str(output_graph), nk.Format.NetworkitBinary)

    to_remove: Set[int] = set()
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    components = list(cc.getComponents())
    random.shuffle(components)
    for component in tqdm(components, desc="Iterating over components..."):
        component = sorted(component)
        to_remove.update(component[1:])

    return to_remove


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

    if args.path:
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
