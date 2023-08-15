#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:37:44
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from collections import Counter


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.

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
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


class RankUnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.
    This version uses rank structure alongside path compression. (Union by Rank)
    Applying either union by rank or path compression results in a time complexity of O( log (n) ) each.
    Applying both further reduces this to O( inverse_ackermann (n) )
    (inverse ackermann is a very slow growing function.)


    Examples
    --------
    >>> ruf = RankUnionFind()
    >>> ruf.union(1, 2)
    >>> ruf.union(2, 3)
    >>> ruf.union(4, 5)
    >>> ruf.find(1)
    1
    >>> ruf.find(2)
    1
    >>> ruf.find(3)
    1
    >>> ruf.find(4)
    4
    >>> ruf.find(5)
    4
    >>> ruf.rank[1]
    1
    >>> ruf.rank[2]
    0
    >>> ruf.union(3, 4)
    >>> ruf.find(1) == ruf.find(5) == 1
    True
    >>> ruf.find(7)
    7
    >>> ruf.rank[7]
    0
    """

    def __init__(self):
        self.parent = {}
        self.rank = Counter()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        # path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)

        # If both elements are already in the same set, do nothing
        # The line in original UnionFind `self.parent[px] = self.parent[py] = min(px, py)` is redundant when px == py
        if px == py:
            return
        # Attach the smaller rank tree under the root of the larger rank tree
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            # If ranks are equal, choose one as the new root and increment its rank
            self.parent[py] = px
            self.rank[px] += 1
