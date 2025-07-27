"""
Python bindings for uf_rush: A lock-free, thread-safe Union-Find implementation.

This module provides a fast, concurrent implementation of the Union-Find (Disjoint Set)
data structure implemented in Rust with Python bindings.
"""

from typing import Generic
from typing import TypeVar

from uf_rush import MAX_SIZE
from uf_rush import UFRush as _UFRush

T = TypeVar("T")


class UFRush:
    """
    A fast, lock-free Union-Find implementation with Python-compatible interface.

    This is a thin wrapper around the Rust implementation that provides
    additional Python conveniences while maintaining high performance.

    Examples
    --------
    >>> uf = UFRush(10)
    >>> uf.union(1, 2)
    True
    >>> uf.union(2, 3)
    True
    >>> uf.same(1, 3)
    True
    >>> uf.find(1) == uf.find(3)
    True
    """

    def __init__(self, size: int):
        """
        Create a new Union-Find structure.

        Args:
            size: Number of elements (0 to size-1)
        """
        self._uf = _UFRush(size)

    def size(self) -> int:
        """Return the number of elements."""
        return self._uf.size()

    def find(self, x: int) -> int:
        """Find the representative of the set containing x."""
        return self._uf.find(x)

    def union(self, x: int, y: int) -> bool:
        """Unite the sets containing x and y."""
        return self._uf.union(x, y)

    def same(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self._uf.same(x, y)

    def clear(self) -> None:
        """Clear the union-find structure."""
        self._uf.clear()

    def connected(self, x: int, y: int) -> bool:
        """Alias for same() method for compatibility."""
        return self.same(x, y)

    def __repr__(self) -> str:
        return f"UFRush(size={self.size()})"

    def __str__(self) -> str:
        return f"UFRush with {self.size()} elements"


class UnionFind(Generic[T]):
    """
    Generic Union-Find implementation that supports arbitrary hashable types.

    This provides a drop-in replacement for the existing Python UnionFind
    with similar API but backed by the fast Rust implementation when possible.

    Examples
    --------
    >>> uf = UnionFind()
    >>> uf.union(1, 2)
    >>> uf.union(2, 3)
    >>> uf.union(4, 5)
    >>> uf.find(1) == uf.find(3)
    True
    >>> uf.find(1) == uf.find(4)
    False
    """

    def __init__(self, max_size: int = MAX_SIZE):
        self._id_to_index: dict[T, int] = {}
        self._index_to_id: dict[int, T] = {}
        self._next_index = 0
        self._max_rust_size = max_size
        self._rust_uf: UFRush = UFRush(self._max_rust_size)

    def _get_or_create_index(self, x: T) -> int:
        """Get or create an integer index for the given element."""
        if x not in self._id_to_index:
            self._id_to_index[x] = self._next_index
            self._index_to_id[self._next_index] = x
            self._next_index += 1

        return self._id_to_index[x]

    def find(self, x: T) -> T:
        """Find the representative of the set containing x."""
        x_idx = self._get_or_create_index(x)
        repr_idx = self._rust_uf.find(x_idx)
        return self._index_to_id[repr_idx]

    def union(self, x: T, y: T) -> None:
        """Unite the sets containing x and y."""
        x_idx = self._get_or_create_index(x)
        y_idx = self._get_or_create_index(y)
        self._rust_uf.union(x_idx, y_idx)
        return

    def same(self, x: T, y: T) -> bool:
        """Check if x and y are in the same set."""
        x_idx = self._get_or_create_index(x)
        y_idx = self._get_or_create_index(y)
        return self._rust_uf.same(x_idx, y_idx)

    def connected(self, x: T, y: T) -> bool:
        """Alias for same() method."""
        return self.same(x, y)

    def reset(self) -> None:
        """Reset the union-find structure."""
        self._id_to_index = {}
        self._index_to_id = {}
        self._next_index = 0
        self._rust_uf.clear()


__all__ = ["MAX_SIZE", "UFRush", "UnionFind"]
