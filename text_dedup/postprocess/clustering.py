#!/usr/bin/env python
# @Date         : 2021-06-05 10:53:26
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
from annoy import AnnoyIndex
from datasketch import MinHash
from datasketch import MinHashLSH
from mpire import WorkerPool as Pool
from tqdm import tqdm

from text_dedup.embedders.base import Fingerprint
from text_dedup.index.simhash_index import SimhashIndex

logger: logging.Logger = logging.getLogger("text_dedup")


def annoy_clustering(
        signatures: List[Fingerprint],
        f: int = 128,
        metric: Literal[
            "angular",
            "euclidean",
            "manhattan",
            "hamming",
            "dot",
        ] = "angular",
        num_trees: int = 64,
        top_k: int = 100,
        distance_threshold: float = 0.5,
        query_signatures: List[Fingerprint] | None = None,
) -> List[List[int]]:
    """Cluster embeddings with annoy.

    Parameters
    ----------
    signatures : List[Fingerprint]
        List of embeddings
    f : int, optional
        Number of the embedding features, by default 128
    metric : str, optional
        Metric for distance measurement, by default "angular"
    num_trees : int, optional
        Number of the trees for annoy to build, by default 64
    top_k : int, optional
        Top k values to be returned by annoy, by default 100
    distance_threshold : float, optional
        Distance threshold, by default 0.5
    query_signatures : List[Fingerprint], optional
        Embedded queries, by default None

    Returns
    -------
    List[int]
        List of neighbors
    """
    t = AnnoyIndex(f, metric)
    for i, v in enumerate(signatures):
        t.add_item(i, v)  # type: ignore
    t.build(num_trees)

    neighbors: List[List[int]] = []

    if query_signatures is None:
        # Find self duplicates
        query_signatures = signatures

    for i, v in enumerate(signatures):
        current: List[int] = []
        for j, dist in zip(
                *t.get_nns_by_vector(  # type: ignore
                    v,
                    top_k,
                    search_k=-1,
                    include_distances=True,
                ),
        ):
            if dist < distance_threshold:
                current.append(j)
        neighbors.append(current[:])

    return neighbors


def lsh_clustering(
        signatures: List[Fingerprint],
        threshold: float = 0.5,
        num_perm: int = 128,
        query_signatures: Optional[List[Fingerprint]] = None,
        storage_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
) -> List[List[int]]:
    """
    Cluster signatures with LSH.

    Parameters
    ----------
    signatures : List[np.ndarray]
        List of signatures
    threshold : float, optional
        Threshold for similarity, by default 0.5
    num_perm : int, optional
        Number of permutations to use, by default 128
    query_signatures : Optional[List[np.ndarray]], optional
        List of query signatures, by default None
    storage_config : Optional[Dict[str, Any]]
        Storage configuration, by default None
    verbose : bool, optional
        Verbose, by default False

    Returns
    -------
    List[List[int]]
        List of neighbors
    """
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
        storage_config=storage_config,
    )
    if lsh.is_empty() and not lsh.keys:
        with lsh.insertion_session() as session:
            for key, minhash in enumerate(tqdm(signatures, desc="Indexing signatures", disable=not verbose)):
                if f"id-{key}" in lsh.keys:
                    continue
                session.insert(
                    f"id-{key}",
                    MinHash(num_perm=num_perm, hashvalues=minhash),
                )

    neighbors: List[List[int]] = []

    if query_signatures is None:
        query_signatures = signatures

    with Pool(os.cpu_count()) as pool:
        neighbors = pool.map(
            lambda signature: [
                int(x.split("-")[1])
                for x in lsh.query(
                    MinHash(num_perm=num_perm, hashvalues=signature),
                )
            ],
            query_signatures,
            progress_bar=verbose,
            progress_bar_options={"desc": "Querying..."},
        )

    return neighbors


def simhash_clustering(
        signatures: List[Fingerprint],
        hamming_distance: int = 3,
        num_blocks: int = 5,
        query_signatures: List[Fingerprint] | None = None,
        storage_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
) -> List[List[int]]:
    """
    Cluster signatures with simhash.

    Parameters
    ----------
    signatures : List[int]
        List of signatures
    hamming_distance : int, optional
        Hamming distance, by default 3
    num_blocks : Optional[int], optional
        Number of blocks, by default 5
    query_signatures : Optional[List[int]], optional
        List of query signatures, by default None
    storage_config : Optional[Dict[str, Any]]
        Storage configuration, by default None
    verbose : bool, optional
        Verbose, by default False

    Returns
    -------
    List[List[int]]
        List of neighbors
    """
    index = SimhashIndex(
        [(i, signature) for i, signature in enumerate(
            tqdm(signatures, disable=not verbose, desc="Loading signatures"))],
        k=hamming_distance,
        b=num_blocks,
        storage_config=storage_config,
    )

    if query_signatures is None:
        query_signatures = signatures

    neighbors: List[List[int]] = []
    with Pool(os.cpu_count()) as pool:
        neighbors = pool.map(
            index.get_near_dups,
            query_signatures,
            progress_bar=verbose,
            progress_bar_options={"desc": "Querying..."},
        )

    return neighbors
