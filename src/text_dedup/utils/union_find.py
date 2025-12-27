"""Union-Find data structure for efficient clustering."""

from __future__ import annotations


class UnionFind:
    """
    A Union-Find (Disjoint Set Union) data structure for efficient clustering.

    This implementation uses path compression and union by rank for optimal performance.
    """

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        """
        Find the root of the set containing x with path compression.

        Parameters
        ----------
        x : int
            The element to find.

        Returns
        -------
        int
            The root of the set containing x.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """
        Union the sets containing x and y.

        Parameters
        ----------
        x : int
            First element.
        y : int
            Second element.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def reset(self) -> None:
        """Reset the Union-Find structure."""
        self.parent.clear()
        self.rank.clear()

    def get_clusters(self) -> dict[int, int]:
        """
        Get the cluster assignment for all elements.

        Returns
        -------
        dict[int, int]
            A mapping from element to its cluster root.
        """
        return {x: self.find(x) for x in self.parent}
