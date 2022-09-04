#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 10:54:44
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
from typing import List


class UF:  # pragma: no cover
    """
    An implementation of union find data structure. It uses weighted quick union by rank with path compression, based on
    https://python-algorithms.readthedocs.io/en/stable/_modules/python_algorithms/basic/union_find.html.
    """

    def __init__(self, N):
        """Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.
        """

        self._id = list(range(N))
        self._count = N
        self._rank = [0] * N

    def find(self, p):
        """Find the set identifier for the item p."""

        id = self._id
        while p != id[p]:
            p = id[p] = id[id[p]]  # Path compression using halving.
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""

        id = self._id
        rank = self._rank

        i = self.find(p)
        j = self.find(q)
        if i == j:
            return

        self._count -= 1
        if rank[i] < rank[j]:
            id[i] = j
        elif rank[i] > rank[j]:
            id[j] = i
        else:
            id[j] = i
            rank[i] += 1

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"


def get_group_indices(neighbors: List[List[int]]) -> List[int]:
    """Based on the nearest neighbors, find the group/cluster index for each element.

    Parameters
    ----------
    neighbors : List[List[int]]
        List of nearest neighbor indices

    Returns
    -------
    List[int]
        List of group indices

    Examples
    --------
    >>> get_group_indices([[0, 1], [0, 2], [1, 2]])
    [0, 0, 0]
    """
    finder = UF(len(neighbors))
    for i, n in enumerate(neighbors):
        for j in n:
            finder.union(i, j)

    return [finder.find(i) for i in range(len(neighbors))]
