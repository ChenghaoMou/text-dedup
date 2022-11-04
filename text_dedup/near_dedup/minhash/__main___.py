#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import gc
import glob
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set

import datasets
import dill as pickle
import networkit as nk
import numpy as np
import typer
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasketch import LeanMinHash
from datasketch import MinHash
from datasketch import MinHashLSH
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
multiprocessing.set_start_method("fork", force=True)


random.seed(42)
MINHASH_SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))
logger.propagate = False
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


def load_dataset_with_config(conf: Dict[str, Any]) -> Dataset:
    """
    Load a dataset based on the configuration. Be careful about changing this function,
    as it is used for caching the intermediate results.

    Parameters
    ----------
    conf : Dict[str, Any]
        The configuration. Mainly, there are three ways to load a dataset:
        1. Directly from th ehub
        2. From a local git repository
        3. From a local dataset directory that was saved by `save_to_disk` before

    Returns
    -------
    Dataset
        The loaded dataset.
    """

    # Load from hub
    if not conf["lfs"]:
        ds = load_dataset(
            conf["dataset"],
            conf["config"],
            data_dir=conf["data_dir"],
            split=conf["split"],
            use_auth_token=True,
            cache_dir=conf["cache_dir"],
        )
    # Or load from git lfs files, if there isn't a local concatenated dataset
    elif not os.path.exists(conf["concat_output"]):
        datasets = []
        # In practice, it might stuck here, you can hit Ctrl+C and run it again.
        for file in tqdm(sorted(glob.glob(conf["data_dir"] + "/*.jsonl")), desc="Loading datasets..."):
            datasets.append(load_dataset("json", data_files=file, split=conf["split"], cache_dir=conf["cache_dir"]))
        ds = concatenate_datasets(datasets)
        ds.save_to_disk(conf["concat_output"])
        ds = load_from_disk(conf["concat_output"])
    # Or load from the concatenated dataset
    else:
        ds = load_from_disk(conf["concat_output"])

    # Assign unique index to each record
    # A token length filtering was used in the Python experiment
    ds = ds.filter(
        lambda x: len({t for t in NON_ALPHA.split(x[conf["column"]]) if t}) >= conf["min_token_length"],
        num_proc=os.cpu_count(),
        desc="Filtering records...",
    )
    ds = ds.map(
        lambda _, idx: {"__id__": idx},
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Adding index...",
    )

    return ds


def embed_func(idx: int, content: str, *, num_perm: int) -> Dict[str, Any]:
    """
    Embed the content of a record into a MinHash object. This function should be
    used with multiprocessing and it scales well with the number of cores.

    Parameters
    ----------
    idx : int
        The index of the record.
    content : str
        The content to embed.
    num_perm : int
        The number of permutations to use in the MinHash object.
    seed : int
        The seed to use in the MinHash object.

    Returns
    -------
    Dict[str, Any]
        The MinHash signature and the index of the record.

    Examples
    --------
    >>> result = embed_func(0, "Hello world!", num_perm=128)
    >>> result["__id__"]
    0
    >>> result["__signature__"].shape
    (128,)
    >>> result["__signature__"].dtype
    dtype('uint64')
    """
    m = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
    m.update_batch([token.encode("utf-8") for token in {t for t in NON_ALPHA.split(content) if t}])
    return {"__signature__": m.hashvalues, "__id__": idx}


def query_func(idx: int, signature: np.ndarray, *, index: MinHashLSH) -> Dict[str, Any]:
    """
    Query the MinHashLSH index for the record. This function can be used with multiprocessing
    as long as the index is shared across processes.

    Parameters
    ----------
    index : MinHashLSH
        The MinHashLSH index. It is shared across all processes when using multiprocessing with fork without copy.
    record : Dict[str, Any]
        The record to query.

    Returns
    -------
    Dict[str, Any]
        The query result.

    Examples
    --------
    >>> data = ["Hello world!", "Hello world"]
    >>> signatures = [embed_func(i, content, num_perm=128) for i, content in enumerate(data)]
    >>> index = MinHashLSH(threshold=0.5, num_perm=128)
    >>> for signature in signatures:
    ...     index.insert(
    ...         signature["__id__"],
    ...         MinHash(num_perm=128, hashvalues=signature["__signature__"], seed=MINHASH_SEED)
    ...     )
    >>> query_func(0, signatures[0]["__signature__"], index=index)
    {'__neighbors__': [1], '__id__': 0}
    >>> query_func(1, signatures[1]["__signature__"], index=index)
    {'__neighbors__': [0], '__id__': 1}
    """
    return {
        "__neighbors__": [
            dup_idx
            for dup_idx in index.query(
                LeanMinHash(seed=MINHASH_SEED, hashvalues=signature),
            )
            if dup_idx != idx  # exclude itself
        ],
        "__id__": idx,
    }


def calculate_average_false_positive_rate(
    clusters: List[List[int]],
    reference_records: Iterable | Dataset,
    threshold: float,
    column: str,
):
    """
    Calculate the average false positive rate within each cluster. The false positives are defined as
    number of examples that have a maximum jaccard similarity with any example in the cluster that is
    less than the threshold. The false positive rate is defined as the number of false positives divided
    by the number of examples in the cluster. The average false positive rate is defined as the average
    of the false positive rate across all clusters given.

    Parameters
    ----------
    clusters : List[List[int]]
        The clusters of duplicate records.
    reference_records : Iterable | Dataset
        The reference records. It can be an iterable or a Dataset.
    threshold : float
        The threshold to use for calculating the false positive rate.
    column : str
        The column to use for calculating the false positive rate.
    """
    cluster_false_positive_rates: List[float] = []
    deltas: List[float] = []

    for cluster in tqdm(clusters, desc="Calculating sampling false positive rate..."):
        num_false_positives = 0
        ids = sorted(cluster)
        for i, x in enumerate(ids):
            is_false_positive = True
            max_similarity = -float("inf")
            for j, y in enumerate(ids):
                if i == j:
                    continue
                # TODO This can be redundant but we only calculate this for a small sample
                similarity = jaccard_similarity(reference_records[x][column], reference_records[y][column])
                max_similarity = max(max_similarity, similarity)
                if max_similarity >= threshold:
                    is_false_positive = False
                    break
            if is_false_positive:
                num_false_positives += 1
                deltas.append(threshold - max_similarity)
        cluster_false_positive_rates.append(num_false_positives / len(ids))

    logger.info(
        f"Average false positive rate from {len(clusters)} clusters: {np.mean(cluster_false_positive_rates):.2f}"
    )
    logger.info(f"Similarity delta stats from threshold:")
    logger.info(f"-  Max : {np.max(deltas):0.2f}")
    logger.info(f"-  Min : {np.min(deltas):0.2f}")
    logger.info(f"-  Mean: {np.mean(deltas):0.2f}")
    logger.info(f"-  Std : {np.std(deltas):0.2f}")


def find_duplicate_communities(
    records: Iterable | Dataset,
    community_detection: bool,
    output: str,
    report_false_positive_rate: bool = False,
    reference_records: Iterable | Dataset | None = None,
    threshold: float = 0.85,
    column: str = "content",
    input_graph: str | None = None,
    output_graph: str | None = None,
    verbose: bool = False,
) -> Set[int]:
    """
    Find the duplicate communities from the queried dataset.

    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains both `__id__` and `__neighbors__`.
    community_detection : bool
        Whether to use community detection to find the duplicate communities, or to use the connected components.
    output : str
        The output file to save the duplicate communities.
    report_false_positive_rate : bool
        Whether to report the false positive rate.
    reference_records : Iterable | Dataset | None
        The reference records. It can be an iterable or a Dataset. It is only used when `report_false_positive_rate` is True.
    threshold : float
        The threshold to use for calculating the false positive rate. It is only used when `report_false_positive_rate` is True.
    column : str
        The column to use for calculating the false positive rate. It is only used when `report_false_positive_rate` is True.
    input_graph : str | None
        The input graph file to load the graph from.
    output_graph : str | None
        The output graph file to save the graph to.
    verbose : bool
        Whether to print verbose logs and false positive rate stats.

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """
    SAMPLE_MIN_SIZE = 10
    SAMPLE_MAX_SIZE = 100
    SAMPLE_SIZE = 10
    if input_graph is not None:
        g = nk.readGraph(str(input_graph), nk.Format.NetworkitBinary)
    else:
        g = nk.graph.Graph()
        for record in tqdm(records, desc="Constructing graph..."):
            for y in record["__neighbors__"]:
                g.addEdge(record["__id__"], y, addMissing=True)

        # Merging subgraphs takes roughly the same time as constructing the graph from scratch
        # def create_graph(idx, neighbors):
        #     g = nk.graph.Graph()
        #     for i, y in zip(idx, neighbors):
        #         for neighbor in y:
        #             g.addEdge(i, neighbor, addMissing=True)
        #     return {'__graph__': [pickle.dumps(g)]}

        # graphs = records.map(
        #     create_graph,
        #     input_columns=["__id__", "__neighbors__"],
        #     remove_columns=["__id__", "__neighbors__"],
        #     batched=True,
        #     batch_size=100_000,
        #     num_proc=os.cpu_count(),
        #     desc="Constructing graph...",
        # )
        # g = nk.graph.Graph()
        # for graph in tqdm(graphs, desc="Merging graphs..."):
        #     nk.graphtools.merge(g, pickle.loads(graph["__graph__"]))
        if output_graph is not None:
            if os.path.exists(output_graph):
                os.remove(output_graph)
            nk.writeGraph(g, str(output_graph), nk.Format.NetworkitBinary)

    to_remove: Set[int] = set()
    samples: List[List[int]] = []
    if not community_detection:
        cc = nk.components.ConnectedComponents(g)
        cc.run()
        partition = cc.getPartition()
        components = list(cc.getComponents())
        random.shuffle(components)
        for component in tqdm(components, desc="Iterating over components..."):
            component = sorted(component)
            to_remove.update(component[1:])
            if len(samples) < SAMPLE_SIZE and SAMPLE_MAX_SIZE > len(component) >= SAMPLE_MIN_SIZE:
                samples.append(component[:])
    else:
        algo = nk.community.PLM(g, refine=False)
        algo.run()
        partition = algo.getPartition()
        communities = list(partition.getSubsetIds())
        random.shuffle(communities)
        # This can be slow if there are many communities
        for i in tqdm(communities, desc="Iterating over communities..."):
            ids = partition.getMembers(i)
            to_remove.update(sorted(ids)[1:])
            if len(samples) < SAMPLE_SIZE and SAMPLE_MAX_SIZE > len(ids) >= SAMPLE_MIN_SIZE:
                samples.append(ids)

    nk.graphio.PartitionWriter().write(partition, str(output))
    if report_false_positive_rate and verbose:
        calculate_average_false_positive_rate(
            samples,
            reference_records,
            threshold,
            column,
        )

    return to_remove


def jaccard_similarity(code1: str, code2: str) -> float:
    """
    Calculate the jaccard similarity between two code snippets.

    Parameters
    ----------
    code1 : str
        The first code snippet.
    code2 : str
        The second code snippet.

    Returns
    -------
    float
        The jaccard similarity between the two code snippets.

    Examples
    --------
    >>> jaccard_similarity("a = 1", "a = 2")
    0.3333333333333333
    >>> jaccard_similarity("a = 1", "a = 1")
    1.0
    """
    tokens1 = set([t for t in NON_ALPHA.split(code1) if t.strip()])
    tokens2 = set([t for t in NON_ALPHA.split(code2) if t.strip()])
    return len(tokens1 & tokens2) / max(1, len(tokens1 | tokens2))


def find_duplicate_non_extremes(
    records: Iterable | Dataset,
    reference_records: Iterable | Dataset,
    column: str,
    threshold: float,
) -> None:
    """
    This is a approximation of what has been used in other script.

    This is slow in this implementation as parallelization requires a global variable
    to hold the dataset to query, which is not implemented in this script.
    """
    g = nk.graph.Graph()
    for record in tqdm(records, desc="Constructing graph..."):
        for y in record["__neighbors__"]:
            g.addEdge(record["__id__"], y, addMissing=True)

    to_remove: Set[int] = set()
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    for component in tqdm(sorted(cc.getComponents(), key=len, reverse=True), desc="Iterating over components..."):
        extremes: Set[int] = set()
        # greedy clustering within each component
        for element1 in tqdm(component, leave=False):
            code1 = reference_records[element1][column]
            for element2 in extremes:
                code2 = reference_records[element2][column]
                if jaccard_similarity(code1, code2) >= threshold:
                    break
            else:
                extremes.add(element1)
        to_remove.update([i for i in component if i not in extremes])

    return to_remove


if __name__ == "__main__":

    def run(
        # Dataset parameters
        dataset: str = typer.Option("codeparrot/codeparrot-clean-valid", help="The dataset to use"),
        config: str = typer.Option("default", help="Dataset config"),
        split: str = typer.Option("train", help="Dataset split"),
        data_dir: str = typer.Option(None, help="Dataset data directory"),
        column: str = typer.Option("content", help="Dataset column"),
        cache_dir: str = typer.Option(".cache", help="Cache directory"),
        # MinHash parameters
        num_perm: int = typer.Option(256, help="Number of permutations"),
        seed: int = typer.Option(42, help="Random seed"),
        threshold: float = typer.Option(0.85, help="Minhash threshold"),
        # IO parameters
        input_neighbor_dataset: str = typer.Option(None, help="Resume from a queried dataset"),
        output_neighbor_dataset: str = typer.Option(None, help="Store a queried dataset"),
        input_graph: str = typer.Option(None, help="Resume from a graph"),
        output_graph: str = typer.Option(None, help="Store a graph"),
        input_duplicate_ids: str = typer.Option(None, help="Resume from computed duplicate ids"),
        output_duplicate_ids: str = typer.Option(None, help="Store computed duplicate ids"),
        output: str = typer.Option(None, help="Store the deduplicated dataset"),
        lfs: bool = typer.Option(False, help="Use LFS files"),
        # Preprocessing parameters
        min_token_length: int = typer.Option(10, help="Minimum token length"),
        # Postprocessing parameters
        community_detection: bool = typer.Option(False, "--community-detection", help="Use community detection"),
        report_false_positive_rate: bool = typer.Option(
            False, "--report-false-positive-rate", help="Report false positive rate based on random samples"
        ),
        # Misc parameters
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    ):
        global lsh
        global dup_ids

        OUTPUT_BASE = Path("results") / dataset / config / (data_dir or "all") / split / column
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

        output_neighbor_dataset = output_neighbor_dataset or (OUTPUT_BASE / "neighbors")
        output_graph = output_graph or (OUTPUT_BASE / "graph.networkit")
        output_duplicate_ids = output_duplicate_ids or (OUTPUT_BASE / "duplicate_ids.json")
        output_concat = OUTPUT_BASE / "concat"
        output_index = OUTPUT_BASE / "index.pkl"
        output_unique_paths = OUTPUT_BASE / "unique_paths.json"
        output_community = OUTPUT_BASE / "community.partition"
        output = output or (OUTPUT_BASE / "deduplicated")

        conf = {
            "cache_dir": cache_dir,
            "num_perm": num_perm,
            "seed": seed,
            "threshold": threshold,
            "dataset": dataset,
            "config": config,
            "data_dir": data_dir,
            "split": split,
            "column": column,
            "verbose": verbose,
            "input_neighbor_dataset": input_neighbor_dataset,
            "input_neighbor_dataset": input_neighbor_dataset,
            "input_graph": input_graph,
            "output_graph": output_graph,
            "input_duplicate_ids": input_duplicate_ids,
            "output_duplicate_ids": output_duplicate_ids,
            "output": output,
            "lfs": lfs,
            "index_output": output_index,
            "concat_output": output_concat,
            "community_detection": community_detection,
            "community_output": output_community,
            "report_false_positive_rate": report_false_positive_rate,
            "min_token_length": min_token_length,
        }

        time_measures = {}

        lsh = MinHashLSH(
            threshold=conf["threshold"],
            num_perm=conf["num_perm"],
        )

        time_measures["load_dataset"] = time.time()
        ds = load_dataset_with_config(conf)
        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]

        DATA_SIZE = len(ds)
        start_time = time.time()

        if not input_neighbor_dataset and not input_duplicate_ids:
            # region: embed
            time_measures["embed"] = time.time()
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={"num_perm": conf["num_perm"]},
                input_columns=["__id__", conf["column"]],
                remove_columns=[conf["column"]],
                num_proc=os.cpu_count(),
                desc=f"Fingerprinting...",
            )
            time_measures["embed"] = time.time() - time_measures["embed"]
            # endregion

            # region: index
            if os.path.exists(output_index):
                time_measures["load_index"] = time.time()
                logger.info(f"Loading index from {output_index}")
                with open(output_index, "rb") as f:
                    lsh = pickle.load(f)
                time_measures["load_index"] = time.time() - time_measures["load_index"]
            else:
                time_measures["create_index"] = time.time()
                with lsh.insertion_session() as session:
                    for data in tqdm(embedded, desc="Indexing signatures..."):
                        if data["__id__"] in lsh:
                            continue
                        session.insert(
                            data["__id__"],
                            LeanMinHash(seed=MINHASH_SEED, hashvalues=data["__signature__"]),
                            check_duplication=False,
                        )
                time_measures["create_index"] = time.time() - time_measures["create_index"]
                time_measures["save_index"] = time.time()
                pickle.dump(lsh, open(output_index, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                time_measures["save_index"] = time.time() - time_measures["save_index"]
            # endregion

            # region: freeze
            # This prevents the index's reference count being modified so it can be shared across processes
            # It will take some time as everything will be copied into a permanent memory space
            time_measures["freeze_memory"] = time.time()
            gc.disable()
            gc.freeze()
            time_measures["freeze_memory"] = time.time() - time_measures["freeze_memory"]
            # endregion

            # region: query
            # Do not use fn_kwargs here, it will be pickled instead.
            time_measures["query"] = time.time()
            queried = embedded.map(
                lambda x, y: query_func(x, y, index=lsh),
                num_proc=os.cpu_count(),
                new_fingerprint=hashlib.md5(pickle.dumps(conf)).hexdigest(),
                input_columns=["__id__", "__signature__"],
                remove_columns=["__signature__"],
                desc=f"Querying...",
            )
            time_measures["query"] = time.time() - time_measures["query"]
            # endregion

            # region: save
            time_measures["save_neighbors"] = time.time()
            queried.save_to_disk(output_neighbor_dataset)
            time_measures["save_neighbors"] = time.time() - time_measures["save_neighbors"]
            # endregion

            # region: unfreeze
            time_measures["unfreeze_memory"] = time.time()
            gc.enable()
            gc.unfreeze()
            time_measures["unfreeze_memory"] = time.time() - time_measures["unfreeze_memory"]
            # endregion
        elif not input_duplicate_ids:
            time_measures["load_neighbors"] = time.time()
            queried = load_from_disk(input_neighbor_dataset)
            time_measures["load_neighbors"] = time.time() - time_measures["load_neighbors"]
        else:
            queried = None

        del lsh
        gc.collect()

        if not input_duplicate_ids:
            # region: clustering
            time_measures["clustering"] = time.time()
            queried = queried.filter(
                lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
            )
            dup_ids = find_duplicate_communities(
                records=queried,
                community_detection=conf["community_detection"],
                output=conf["community_output"],
                report_false_positive_rate=conf["report_false_positive_rate"],
                reference_records=ds,
                threshold=conf["threshold"],
                column=conf["column"],
                input_graph=conf["input_graph"],
                output_graph=conf["output_graph"],
            )
            time_measures["clustering"] = time.time() - time_measures["clustering"]
            # endregion

            # region: save
            time_measures["save_duplicate_ids"] = time.time()
            with open(output_duplicate_ids, "w") as f:
                json.dump(list(map(str, dup_ids)), f)
            time_measures["save_duplicate_ids"] = time.time() - time_measures["save_duplicate_ids"]
            # endregion
        else:
            # region: load
            time_measures["load_duplicate_ids"] = time.time()
            with open(input_duplicate_ids, "r") as f:
                dup_ids = set((map(int, json.load(f))))
            time_measures["load_duplicate_ids"] = time.time() - time_measures["load_duplicate_ids"]
            # endregion

        del queried
        gc.collect()

        time_measures["total_processing_time"] = time.time() - start_time

        # region: deduplicate
        # Reload the original dataset
        # ds = load_dataset_with_config(conf)
        time_measures["deduplicate"] = time.time()
        final_data = ds.filter(
            lambda idx: idx not in dup_ids,
            input_columns=["__id__"],
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )
        time_measures["deduplicate"] = time.time() - time_measures["deduplicate"]

        if "repository_name" in final_data.features and "path" in final_data.features:
            time_measures["save_unique_paths"] = time.time()
            with open(output_unique_paths, "w") as f:
                temp = final_data.map(
                    lambda x: {"url": x["repository_name"] + "/" + x["path"]},
                    num_proc=os.cpu_count(),
                    remove_columns=final_data.column_names,
                    desc="Saving unique paths...",
                )
                json.dump(list(set(temp["url"])), f)
            time_measures["save_unique_paths"] = time.time() - time_measures["save_unique_paths"]

        time_measures["save_deduplicated"] = time.time()
        final_data.save_to_disk(output)
        time_measures["save_deduplicated"] = time.time() - time_measures["save_deduplicated"]
        # endregion

        FINAL_DATA_SIZE = len(final_data)
        DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
        LAN = (data_dir or "all").split("/")[-1]
        for key in time_measures:
            logger.info(f"{' '.join(key.split('_')).title():<30}: {time_measures[key]:.2f} seconds")

        logger.info(f"{'Language':<30}: {LAN}")
        logger.info(f"{'Data Number':<30}: {DATA_SIZE}")
        logger.info(f"{'Duplicate Number':<30}: {DUP_SIZE}")
        logger.info(f"{'Duplicate Rate':<30}: {DUP_SIZE / DATA_SIZE:.2%}")
        logger.info(f"{'Total Time':<30}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Output Base':<30}: {OUTPUT_BASE}")
        logger.info(f"{'Concatenated Dataset':<30}: {output_concat}")
        logger.info(f"{'Index':<30}: {output_index}")
        logger.info(f"{'Neighbor Dataset':<30}: {output_neighbor_dataset}")
        logger.info(f"{'Duplicate IDs':<30}: {output_duplicate_ids}")
        logger.info(f"{'Unique Paths':<30}: {output_unique_paths}")
        logger.info(f"{'Graph':<30}: {output_graph}")
        logger.info(f"{'Community':<30}: {output_community}")
        logger.info(f"{'Output':<30}: {output}")
        logger.info("🤗 Happy Deduplicating 🤗")

    typer.run(run)
