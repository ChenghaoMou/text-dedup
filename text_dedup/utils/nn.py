#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 10:53:26
# @Author       : Chenghao Mou (mouchenghao@gmail.com)


from typing import List

import numpy as np
from annoy import AnnoyIndex


def annoy_clustering(
    embeddings: List[np.ndarray],
    f: int,
    metric: str = "angular",
    num_trees: int = 64,
    top_k: int = 100,
    distance_threshold: float = 0.5,
) -> List[List[int]]:
    """Cluster embeddings with annoy.

    Parameters
    ----------
    embeddings : List[np.ndarray]
        List of embeddings
    f : int
        Number of the embedding features
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

    for i, v in enumerate(embeddings):
        current: List[int] = []
        for j, dist in zip(
            *t.get_nns_by_vector(v, top_k, search_k=-1, include_distances=True)
        ):
            if dist < distance_threshold:
                current.append(j)
        neighbors.append(current[:])

    return neighbors