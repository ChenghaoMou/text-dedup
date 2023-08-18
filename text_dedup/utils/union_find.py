#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:37:44
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from collections import Counter


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.
    This version uses both rank structure (Union by Rank) and path compression.
    Applying either union by rank or path compression results in a time complexity of O( log (n) ) each.
    Applying both further reduces this to O( inverse_ackermann (n) )
    (inverse ackermann is a very slow growing function.)


    Examples
    --------
    >>> uf = UnionFind()
    >>> uf.union(1, 2)
    >>> uf.union(2, 3)
    >>> uf.union(4, 5)
    >>> uf.find(1)
    1
    >>> uf.find(2)
    1
    >>> uf.find(3)
    1
    >>> uf.find(4)
    4
    >>> uf.find(5)
    4
    >>> uf.rank[1]
    1
    >>> uf.rank[2]
    0
    >>> uf.union(3, 4)
    >>> uf.find(1) == uf.find(5) == 1
    True
    >>> uf.find(7)
    7
    >>> uf.rank[7]
    0
    """

    def __init__(self):
        self.parent = {}
        # Counter is a subclass of dict with slightly different python and c implementations
        # you can think of it as an optimized defaultdict(int)
        self.rank = Counter()

    def find(self, x):
        try:
            # path compression
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
        except KeyError:
            # KeyError happens if x not in parent
            self.parent[x] = x
        finally:
            return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)

        # If both elements are already in the same set, do nothing
        # The line in original UnionFind `self.parent[px] = self.parent[py] = min(px, py)` is redundant when px == py
        if px == py:
            return

        if self.rank[px] == self.rank[py]:
            # If ranks are equal, choose one as the new root and increment its rank
            # with few duplicates this is likely to be the most common case
            self.parent[py] = px
            self.rank[px] += 1
        # otherwise, assume that leftside is more likely to be higher rank
        # Attach the smaller rank tree under the root of the larger rank tree
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
