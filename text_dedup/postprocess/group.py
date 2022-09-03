#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 10:54:44
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
from typing import List


class UF:  # pragma: no cover
    """An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.

    Source: https://python-algorithms.readthedocs.io/en/stable/_modules/python_algorithms/basic/union_find.html

    An union find data structure can keep track of a set of elements into a number
    of disjoint (nonoverlapping) subsets. That is why it is also known as the
    disjoint set data structure. Mainly two useful operations on such a data
    structure can be performed. A *find* operation determines which subset a
    particular element is in. This can be used for determining if two
    elements are in the same subset. An *union* Join two subsets into a
    single subset.

    The complexity of these two operations depend on the particular implementation.
    It is possible to achieve constant time (O(1)) for any one of those operations
    while the operation is penalized. A balance between the complexities of these
    two operations is desirable and achievable following two enhancements:

    1.  Using union by rank -- always attach the smaller tree to the root of the
        larger tree.
    2.  Using path compression -- flattening the structure of the tree whenever
        find is used on it.

    complexity:
        * find -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
        `inverse ackerman function
        <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.
        * union -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
        `inverse ackerman function
        <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.
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
