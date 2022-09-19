from __future__ import annotations

from collections import deque
from typing import Deque
from typing import Generator
from typing import List
from typing import Literal
from typing import Sequence


def merge_intervals(
        intervals: List[slice],
        merge_strategy: Literal["longest", "overlapping"] = "longest",
) -> List[slice]:
    """
    Merge overlapping intervals.

    Parameters
    ----------
    intervals : List[slice]
        List of intervals
    merge_strategy : Literal["longest", "overlapping"]
        Strategy to merge intervals, by default "longest"
        "overlapping": merge overlapping intervals
        "longest": only ignore duplicate substrings, this is useful because
        when [2, 4] and [3, 5] are duplicates, [2, 5] might be not

    Returns
    -------
    List[slice]
        List of merged intervals

    Examples
    --------
    >>> merge_intervals([slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), slice(3, 13, None),
    ... slice(4, 14, None), slice(5, 15, None), slice(6, 16, None), slice(7, 21, None)], merge_strategy='overlapping')
    [slice(0, 21, None)]
    >>> merge_intervals([slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), slice(3, 13, None),
    ... slice(4, 14, None), slice(5, 15, None), slice(6, 16, None), slice(7, 21, None)],
    ... merge_strategy='longest') #doctest: +ELLIPSIS
    [slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), ... slice(7, 21, None)]
    >>> merge_intervals([slice(0, 2), slice(2, 4), slice(4, 5)], 'overlapping')
    [slice(0, 5, None)]
    >>> merge_intervals([slice(0, 4), slice(2, 4), slice(4, 5)], 'longest')
    [slice(0, 4, None), slice(4, 5, None)]
    >>> merge_intervals([slice(0, 10, None), slice(0, 10, None), slice(0, 10, None), slice(0, 10, None), slice(0, 10, None)])
    [slice(0, 10, None)]
    """
    if len(intervals) == 0:
        return []

    q = deque(sorted(
        list(
            map(
                lambda s: slice(s[0], s[1]),
                {(s.start, s.stop) for s in intervals},
            ),
        ),
        key=lambda x: (x.start, -x.stop),
    ))

    merged: List[slice] = []

    while q:
        current = q.popleft()

        if not merged:
            merged.append(current)
            continue

        prev = merged[-1]
        if merge_strategy == "overlapping":
            if prev.stop >= current.start:
                merged[-1] = slice(prev.start, max(prev.stop, current.stop))
            else:
                merged.append(current)
        elif merge_strategy == "longest":
            if current.stop <= prev.stop:  # ignore substrings
                continue
            else:
                merged.append(current)

    return merged


def restore(
        boundaries: Sequence[slice], segments: str | Sequence[slice],
) -> Generator:
    """
    Restore the duplicate slices from seg_file to their original document boundaries.

    Parameters
    ----------
    boundaries : List[slice]
        List of slices document boundary offsets.
    segments: Union[str, List[slice]]
        Path to the segmented file with duplicate offsets or a list of duplicate slices.

    Yields
    ------
    int, slice
        index and offset

    Examples
    --------
    >>> list(restore(
    ... [slice(0, 10, None),
    ... slice(10, 20, None)],
    ... [slice(0, 5, None),
    ... slice(5, 10, None),
    ... slice(5, 15, None),
    ... slice(5, 19, None)]))
    [(0, slice(0, 5, None)), (0, slice(5, 10, None)), (1, slice(0, 5, None)), (1, slice(0, 9, None))]
    """
    indices: Deque[slice] = deque([])

    if isinstance(segments, str):
        with open(segments) as f:
            for line in f:
                try:
                    left, right = line.strip().split(' ', 1)
                    if left.isdigit() and right.isdigit():
                        indices.append(slice(int(left), int(right)))
                except ValueError:
                    continue
    else:
        indices = deque(segments)

    for i, s in enumerate(boundaries):
        while indices:
            curr_slice = indices.popleft()
            while curr_slice.stop <= s.start and indices:
                curr_slice = indices.popleft()

            x, y = curr_slice.start, curr_slice.stop
            if y <= s.start:
                break

            if x >= s.stop:
                indices.appendleft(slice(x, y))
                break

            if s.start <= x < s.stop <= y:
                yield i, slice(x - s.start, s.stop - s.start)
                if y > s.stop:
                    indices.appendleft(slice(s.stop, y))
                break
            elif s.start <= x < y <= s.stop:
                yield i, slice(x - s.start, y - s.start)
                continue
            elif x < s.start < y <= s.stop:
                yield i, slice(0, y - s.start)
                continue
            elif x < s.start < s.stop <= y:
                yield i, slice(0, s.stop - s.start)
                if y > s.stop:
                    indices.appendleft(slice(s.stop, y))
                break


def restore_and_merge(
        boundaries: Sequence[slice],
        segments: str | Sequence[slice],
        k: int,
        merge_strategy: Literal["longest", "overlapping"] = "longest",
) -> List[List[slice]]:
    """
    Restore the duplicate slices from seg_file to their original document boundaries and merge them.

    Parameters
    ----------
    boundaries : List[slice]
        List of slices document boundary offsets.
    segments: Union[str, List[slice]]
        Path to the segmented file with duplicate offsets or a list of duplicate slices.
    k : int
        Minimum substring length to be considered as a duplicate.
    merge_strategy : Literal["longest", "overlapping"]
        Strategy to merge intervals, by default "longest".

    Returns
    -------
    List[List[slice]]
        List of duplicate substring slices.

    Examples
    --------
    >>> restore_and_merge(
    ... [slice(0, 10, None),
    ... slice(10, 20, None)],
    ... [slice(0, 5, None),
    ... slice(5, 10, None),
    ... slice(5, 15, None),
    ... slice(5, 19, None)],
    ... 5)
    [[slice(0, 5, None), slice(5, 10, None)], [slice(0, 9, None)]]
    >>> restore_and_merge(
    ... [slice(0, 10, None),
    ... slice(10, 20, None)],
    ... [slice(0, 5, None),
    ... slice(5, 10, None),
    ... slice(5, 15, None),
    ... slice(5, 19, None)],
    ... 5, 'overlapping')
    [[slice(0, 10, None)], [slice(0, 9, None)]]
    """
    results: List[List[slice]] = [[] for _ in boundaries]
    for idx, s in restore(boundaries, segments):
        if s.stop - s.start >= k:
            results[int(idx)].append(s)
    for i in range(len(results)):
        results[i] = merge_intervals(results[i], merge_strategy)
    return results
