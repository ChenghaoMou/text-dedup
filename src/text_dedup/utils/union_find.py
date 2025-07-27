#!/usr/bin/env python
# @Date    : 2022-12-26 15:37:44
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from collections import Counter
from typing import Generic

from typing_extensions import TypeVar

T = TypeVar("T")


class UnionFind(Generic[T]):
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

    def __init__(self) -> None:
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = Counter()

    def find(self, x: T) -> T:
        try:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
        except KeyError:
            self.parent[x] = x

        return self.parent[x]

    def union(self, x: T, y: T) -> None:
        px = self.find(x)
        py = self.find(y)

        if px == py:
            return

        if self.rank[px] == self.rank[py]:
            self.parent[py] = px
            self.rank[px] += 1
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py

    def reset(self) -> None:
        self.parent = {}
        self.rank = Counter()
