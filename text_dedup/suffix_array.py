#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 11:48:38
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from collections import deque
from pathlib import Path
from pathlib import PosixPath
from typing import Deque
from typing import Generator
from typing import List
from typing import Literal
from typing import Sequence
from typing import Tuple

import datasets
from datasets import Dataset
from datasets import load_dataset

from text_dedup import logger
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils import add_sa_args
from text_dedup.utils.timer import Timer

random.seed(42)
datasets.logging.set_verbosity_error()


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

    q = deque(
        sorted(
            list(
                map(
                    lambda s: slice(s[0], s[1]),
                    {(s.start, s.stop) for s in intervals},
                ),
            ),
            key=lambda x: (x.start, -x.stop),
        )
    )

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
    boundaries: Sequence[slice],
    segments: str | Path | Sequence[slice],
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

    if isinstance(segments, (str, PosixPath)):
        with open(segments) as f:
            for line in f:
                try:
                    left, right = line.strip().split(" ", 1)
                    if left.isdigit() and right.isdigit():
                        indices.append(slice(int(left), int(right)))
                except ValueError:
                    continue
    elif isinstance(segments, list):
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
    segments: str | Path | Sequence[slice],
    k: int,
    merge_strategy: Literal["longest", "overlapping"] = "longest",
) -> Tuple[List[List[slice]], int]:
    """
    Restore the duplicate slices from seg_file to their original document boundaries and merge them.

    Parameters
    ----------
    boundaries : List[slice]
        List of slices document boundary offsets.
    segments: Union[str, List[slice]]
        Path to the segmented file with duplicate offsets or a list of duplicate slices.
    k : int
        The minimum duplicate substring byte length.
    merge_strategy : Literal["longest", "overlapping"], optional
        The merge strategy to use, by default "longest"

    Returns
    -------
    Tuple[List[List[slice]], int]
        List of merged slices for each document and the duplicate size.

    Examples
    --------
    >>> restore_and_merge(
    ... [slice(0, 10, None), slice(10, 20, None)],
    ... [slice(0, 5, None), slice(5, 10, None), slice(12, 19, None)],
    ... 5, 'longest')
    ([[slice(0, 5, None), slice(5, 10, None)], [slice(2, 9, None)]], 17)
    >>> restore_and_merge(
    ... [slice(0, 10, None), slice(10, 20, None)],
    ... [slice(0, 5, None), slice(5, 10, None), slice(12, 19, None)],
    ... 5, 'overlapping')
    ([[slice(0, 10, None)], [slice(2, 9, None)]], 17)
    """
    duplicate_size = 0
    results: List[List[slice]] = [[] for _ in boundaries]
    for idx, s in restore(boundaries, segments):
        if s.stop - s.start >= k:
            results[int(idx)].append(s)
    for i in range(len(results)):
        results[i] = merge_intervals(results[i], merge_strategy)
        duplicate_size += sum([s.stop - s.start for s in results[i]])
    return results, duplicate_size


def __run_command(cmd: str, cwd: str):
    p = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
    )
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command {cmd} failed with code {code}. CWD: {cwd}")


def clean_up(text: str, slices: List[slice]) -> str:
    """
    Remove duplicate substrings from the text.

    Parameters
    ----------
    text : str
        Text to remove duplicate substrings from.
    slices : List[slice]
        List of slices to remove.

    Returns
    -------
    str
        Text with duplicate substrings removed.

    Examples
    --------
    >>> clean_up("This is a test.", [slice(0, 4, None), slice(5, 7, None)])
    '  a test.'
    """
    byte_array = bytearray(text, "utf-8")
    result = bytearray()
    start = 0
    for s in slices:
        result.extend(byte_array[start : s.start])
        start = s.stop
    result.extend(byte_array[start:])

    return result.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="text-dedup.suffixarray",
        description="Deduplicate text using Suffix Array Deduplication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_sa_args(parser)

    args = parser.parse_args()

    assert args.path is not None, "Please specify `path` for `load_dataset`."

    temp_output_dir = Path(args.google_repo_path) / "output"
    temp_dir = Path(args.google_repo_path) / "tmp"
    temp_output_dir.mkdir(exist_ok=True, parents=True)
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_text = "output/temp_text.txt"
    temp_output = "output/temp_output.txt"
    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds: Dataset = load_dataset(  # type: ignore
                path=args.path,
                name=args.name,
                data_dir=args.data_dir,
                data_files=args.data_files,
                split=args.split,
                revision=args.revision,
                cache_dir=args.cache_dir,
                token=args.use_auth_token,
            )

        with timer("Preprocessing"):
            offsets: List[slice] = []
            start = 0
            with open(Path(args.google_repo_path) / temp_text, "wb") as f:
                for doc in ds:
                    doc_bytes = doc[args.column].encode("utf-8")
                    end = start + len(doc_bytes)
                    offsets.append(slice(start, end))
                    start = end
                    f.write(doc_bytes)

        with timer("SuffixArray"):
            __run_command(
                f"python scripts/make_suffix_array.py {temp_text}",
                args.google_repo_path,
            )

        with timer("SelfSimilar"):
            __run_command(
                f"cargo run self-similar --data-file {temp_text}"
                f" --length-threshold {args.k} --cache-dir {args.cache_dir} --num-threads {args.num_proc}",
                args.google_repo_path,
            )
            __run_command(
                f"cargo run collect --data-file {temp_text}"
                f" --length-threshold {args.k} --cache-dir {args.cache_dir} >"
                f" {temp_output}",
                args.google_repo_path,
            )

        with timer("Restore"):
            duplicate_slices, duplicate_size = restore_and_merge(
                offsets,
                Path(args.google_repo_path) / temp_output,
                args.k,
                args.strategy,
            )

        with timer("Deduplicate"):
            ds = ds.map(
                lambda content, idx: {
                    args.column: clean_up(content, duplicate_slices[idx]),
                },
                with_indices=True,
                input_columns=[args.column],
                desc="Deduplicating",
            ).filter(
                lambda content: len(content) > 0,
                input_columns=[args.column],
                desc="Filtering empty documents",
            )

        with timer("Saving"):
            ds.save_to_disk(args.output)

        with timer("Cleaning"):
            if args.clean_cache:
                ds.cleanup_cache_files()
                shutil.rmtree(temp_output_dir)
                shutil.rmtree(temp_dir)
                shutil.rmtree(args.cache_dir)

    PAD = 30
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f} seconds")

    logger.info(f"{'Before':<{PAD}}: {start} bytes ({len(offsets)})")
    logger.info(f"{'After':<{PAD}}: {start - duplicate_size} bytes ({len(ds)})")
