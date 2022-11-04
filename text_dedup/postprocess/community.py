#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/2/22
# description : Detecting communities in the graph of near-duplicate graph.

from typing import List
from typing import Set
from typing import Tuple

import networkx as nx


def construct_graph(pairs: List[Tuple[int, List[int]]]) -> nx.Graph:
    g = nx.Graph()
    for x, points in pairs:
        for y in points:
            g.add_edge(x, y)

    return g


def find_communities(g: nx.Graph) -> List[List[int]]:
    communities = []
    for c in nx.connected_components(g):
        communities.append(list(c))

    return communities


def get_remove_list(pairs: List[Tuple[int, List[int]]]) -> Set[int]:
    g = construct_graph(pairs)
    communities = find_communities(g)
    to_remove = set()
    for community in communities:
        to_remove.update(community[1:])
    return to_remove
