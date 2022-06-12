from __future__ import annotations

from collections import deque
from typing import Deque, Generator, List


def restore(
    offsets: List[tuple[int, int]], seg_file: str,
) -> Generator:
    """
    Restore the original text from the offsets.

    Parameters
    ----------
    offsets : List[Tuple[int, int]]
        List of (start, end) offsets.
    seg_file : str
        Path to the segmented file with duplicate offsets.

    Yields
    ------
    int, Tuple[int, int]
        index, (start, end) offset
    """
    indices: Deque[tuple[int, int]] = deque([])
    with open(seg_file) as f:
        for line in f:
            try:
                left, right = line.strip().split(' ', 1)
                if left.isdigit() and right.isdigit():
                    indices.append((int(left), int(right)))
            except Exception:
                pass

    for i, (start, end) in enumerate(offsets):
        while indices:
            x, y = indices.popleft()
            while y <= start and indices:
                x, y = indices.popleft()

            if y <= start:
                break

            if x >= end:
                indices.appendleft((x, y))
                break

            if start <= x < end <= y:
                yield i, (x - start, end - start)
                if y > end:
                    indices.appendleft((end, y))
                break
            elif start <= x < y <= end:
                yield i, (x - start, y - start)
                continue
            elif x < start < y <= end:
                yield i, (0, y - start)
                continue
            elif x < start < end <= y:
                yield i, (0, end - start)
                if y > end:
                    indices.appendleft((end, y))
                break
