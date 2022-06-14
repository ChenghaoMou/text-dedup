#!/usr/bin/env python
# @Date         : 2021-06-05 10:53:26
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional

import numpy as np
from annoy import AnnoyIndex
from datasketch import MinHash, MinHashLSH
from multiprocess import Pool
from tqdm import tqdm

from text_dedup.utils.simhash_index import SimhashIndex

logger: logging.Logger = logging.getLogger('text_dedup')


def annoy_clustering(
    embeddings: List[np.ndarray],
    f: int = 128,
    metric: Literal[
        'angular',
        'euclidean',
        'manhattan',
        'hamming',
        'dot',
    ] = 'angular',
    num_trees: int = 64,
    top_k: int = 100,
    distance_threshold: float = 0.5,
    query_embeddings: List[np.ndarray] | None = None,
) -> List[List[int]]:
    """Cluster embeddings with annoy.

    Parameters
    ----------
    embeddings : List[np.ndarray]
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

    Returns
    -------
    List[int]
        List of neighbors
    """
    t = AnnoyIndex(f, metric)
    for i, v in enumerate(embeddings):
        t.add_item(i, v)
    t.build(num_trees)

    neighbors: List[List[int]] = []

    if query_embeddings is None:
        query_embeddings = embeddings

    for i, v in enumerate(embeddings):
        current: List[int] = []
        for j, dist in zip(
            *t.get_nns_by_vector(
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
    signatures: List[np.ndarray],
    threshold: float = 0.5,
    num_perm: int = 128,
    query_signatures: Optional[List[np.ndarray]] = None,
    redis_basename: Optional[str] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    skip_indexing_if_exists: bool = False,
) -> List[List[int]]:
    """
    Cluster embeddings with LSH.

    Parameters
    ----------
    signatures : List[np.ndarray]
        List of embeddings
    threshold : float, optional
        Threshold for similarity, by default 0.5
    num_perm : int, optional
        Number of permutations to use, by default 128
    query_signatures : Optional[List[np.ndarray]], optional
        List of query embeddings, by default None
    redis_basename : Optional[str], optional
        Redis basename, by default None
    redis_host : Optional[str], optional
        Redis host, by default None
    redis_port : Optional[int], optional
        Redis port, by default None
    skip_indexing_if_exists : bool, optional
        Skip indexing if exists, by default False

    Returns
    -------
    List[List[int]]
        List of neighbors
    """
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
        storage_config={
            'type': 'redis',
            'basename': redis_basename.encode('utf-8'),
            'redis': {'host': redis_host, 'port': redis_port},
        }
        if redis_basename is not None
        else None,
    )
    if lsh.is_empty() or not skip_indexing_if_exists:
        with lsh.insertion_session() as session:
            for key, minhash in enumerate(signatures):
                session.insert(
                    f'id-{key}',
                    MinHash(num_perm=num_perm, hashvalues=minhash),
                )
        logger.debug(
            f'LSH index already exists (size: {lsh.get_counts()}), skipped indexing'
        )

    neighbors: List[List[int]] = []

    if query_signatures is None:
        query_signatures = signatures

    with Pool(os.cpu_count()) as pool:
        neighbors = pool.map(
            lambda signature: [
                int(x.split('-')[1])
                for x in lsh.query(
                    MinHash(num_perm=num_perm, hashvalues=signature),
                )
            ],
            tqdm(query_signatures, desc='Querying...'),
        )

    return neighbors


def simhash_clustering(
    signatures: List[int],
    hamming_distance: int = 3,
    num_blocks: int = 5,
    query_signatures: List[int] | None = None,
) -> List[List[int]]:
    """
    Cluster embeddings with simhash.

    Parameters
    ----------
    signatures : List[int]
        List of embeddings
    hamming_distance : int, optional
        Hamming distance, by default 3
    num_blocks : Optional[int], optional
        Number of blocks, by default 5
    query_signatures : Optional[List[int]], optional
        List of query embeddings, by default None
    skip_indexing_if_exists : bool, optional
        Skip indexing if exists, by default False

    Returns
    -------
    List[List[int]]
        List of neighbors
    """
    index = SimhashIndex(
        [(i, signature) for i, signature in enumerate(signatures)],
        k=hamming_distance,
        b=num_blocks,
    )

    if query_signatures is None:
        query_signatures = signatures

    neighbors: List[List[int]] = []
    with Pool(os.cpu_count()) as pool:
        neighbors = pool.map(
            lambda signature: list(
                map(int, index.get_near_dups(signature)),
            ),
            tqdm(query_signatures, desc='Querying...'),
        )

    return neighbors


if __name__ == '__main__':

    print(simhash_clustering([1, 1024, 1231241, 1, 1025]))
