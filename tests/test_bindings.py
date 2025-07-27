#!/usr/bin/env python3

from text_dedup.utils.union_find_rust import MAX_SIZE
from text_dedup.utils.union_find_rust import UFRush
from text_dedup.utils.union_find_rust import UnionFind


def test_rust_uf_rush() -> None:
    """Test the direct Rust UFRush implementation."""
    print("Testing UFRush Rust implementation...")

    # Test basic functionality
    uf = UFRush(10)
    print(f"Size: {uf.size()}")
    assert uf.size() == 10

    # Test union operations
    assert uf.union(1, 2)
    assert uf.union(2, 3)
    assert not uf.union(1, 2)

    # Test same operation
    assert uf.same(1, 3)
    assert not uf.same(1, 4)

    # Test find operation
    assert uf.find(1) == uf.find(2) == uf.find(3)
    assert uf.find(4) != uf.find(1)

    print("✓ UFRush tests passed!")


def test_generic_union_find() -> None:
    """Test the generic UnionFind wrapper."""
    print("Testing UnionFind generic wrapper...")

    uf = UnionFind[str | int](max_size=10)

    # Test with strings
    uf.union("hello", "world")
    uf.union("world", "test")
    uf.union("foo", "bar")

    assert uf.same("hello", "test")
    assert not uf.same("hello", "foo")
    assert uf.find("hello") == uf.find("world")

    # Test with integers
    uf.union(100, 200)
    uf.union(200, 300)

    assert uf.same(100, 300)
    assert not uf.same(100, "hello")

    print("✓ UnionFind tests passed!")


def test_performance_comparison() -> None:
    """Basic performance comparison."""
    import time

    print("Testing performance...")

    size = 10000
    operations = size // 2

    # Test Rust implementation
    start = time.time()
    uf_rust = UFRush(size)
    for i in range(operations):
        uf_rust.union(i, (i + 1) % size)
    rust_time = time.time() - start

    # Test Python fallback
    start = time.time()
    uf_py = UnionFind[int](max_size=size)
    for i in range(operations):
        uf_py.union(i, (i + 1) % size)
    py_time = time.time() - start

    print(f"Rust UFRush: {rust_time:.4f}s")
    print(f"Python UnionFind: {py_time:.4f}s")
    print(f"Speedup: {py_time / rust_time:.2f}x")


if __name__ == "__main__":
    print(f"MAX_SIZE: {MAX_SIZE}")
    test_rust_uf_rush()
    test_generic_union_find()
    test_performance_comparison()
    print("All tests passed! 🎉")
