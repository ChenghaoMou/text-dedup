#!/usr/bin/env python
# @Date    : 2022-04-02 12:15:08
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import logging
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from tqdm import tqdm

from text_dedup.postprocess.suffix_restore import restore
from text_dedup.preprocess.suffix_array import construct_sa

logger = logging.getLogger("text-dedup")


def _merge_intervals(
    slices: List[slice],
    merge_strategy: str = "overlapping",
) -> List[slice]:
    """Merge overlapping intervals.

    Parameters
    ----------
    slices : List[slice]
        List of intervals
    merge_strategy : str, optional
        Strategy to merge intervals, by default "overlapping"
        "overlapping": merge overlapping intervals
        "longest": only ignore duplicate substrings, this is useful because
        the merged overlapping intervals may not be duplicates

    Returns
    -------
    List[slice]
        List of merged intervals

    Examples
    --------
    >>> _merge_intervals([slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), slice(3, 13, None), slice(4, 14, None), slice(5, 15, None), slice(6, 16, None), slice(7, 21, None)], merge_strategy='overlapping')
    [slice(0, 21, None)]
    >>> _merge_intervals([slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), slice(3, 13, None), slice(4, 14, None), slice(5, 15, None), slice(6, 16, None), slice(7, 21, None)], merge_strategy='longest')
    [slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), slice(3, 13, None), slice(4, 14, None), slice(5, 15, None), slice(6, 16, None), slice(7, 21, None)]
    >>> _merge_intervals([slice(0, 2), slice(2, 4), slice(4, 5)], 'overlapping')
    [slice(0, 5, None)]
    >>> _merge_intervals([slice(0, 4), slice(2, 4), slice(4, 5)], 'longest')
    [slice(0, 4, None), slice(4, 5, None)]
    """
    if len(slices) == 0:
        return []

    slices = sorted(
        list(
            map(
                lambda s: slice(s[0], s[1]),
                {(s.start, s.stop) for s in slices},
            ),
        ),
        key=lambda x: (x.start, -x.stop),
    )

    merged: List[slice] = []
    q = deque(slices)

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


@dataclass
class SuffixArrayEmbedder:

    """
    Find duplicate *byte* slices using suffix array.

    Parameters
    ----------
    k: int
        Minimum length of the byte slices.

    Examples
    --------
    >>> from text_dedup.embedders.suffix import SuffixArrayEmbedder
    >>> embedder = SuffixArrayEmbedder(k=100)
    """

    k: int = 100

    def embed(
        self,
        corpus: List[str],
        merge: bool = False,
        merge_strategy: str = "longest",
    ) -> List[List[slice]]:
        """
        Find duplicate byte slices using suffix array.

        Parameters
        ----------
        corpus: List[str]
            List of documents.
        merge: bool
            Whether to merge overlapping intervals.
        merge_strategy: str
            Merge strategy.

        Returns
        -------
        List[List[slice]]
            List of duplicate byte slices.
        """

        assert merge_strategy in ["longest", "overlapping"], f"Invalid merge strategy: {merge_strategy}"

        # This means we need enough memory to store the string
        string = "".join(corpus)
        # This is useful for reconstructing the offsets
        lengths = [len(s) for s in corpus]

        # Construct suffix array
        sa = construct_sa(string)
        slices = []

        # Neighboring suffixes share the same prefix
        # This is where we find the duplicate byte slices
        for x, y in tqdm(
            zip(sa[:-1], sa[1:]),
            total=len(sa) - 1,
            desc="Suffix array querying",
        ):
            # We only care about the suffixes that are longer than k
            matched_length = 0
            while (
                x + matched_length < len(string)
                and y + matched_length < len(string)
                and string[x + matched_length] == string[y + matched_length]
            ):
                matched_length += 1
            if matched_length >= self.k:
                slices.append(slice(x, x + matched_length))
                slices.append(slice(y, y + matched_length))

        q = deque(sorted(slices, key=lambda x: x.start))

        # Make sure the slices are respecting the document boundaries
        start = 0
        ans: List[List[slice]] = []
        for _, length in enumerate(lengths):
            end = start + length
            curr: List[slice] = []
            while q and q[0].start < end:
                s = q.popleft()
                if s.start < start:
                    continue
                if s.stop > end:
                    if end - s.start >= self.k:
                        curr.append(slice(s.start, end))

                    if s.stop - end >= self.k:
                        q.appendleft(slice(end, s.stop))

                if s.stop <= end:
                    curr.append(s)
            if merge:
                ans.append(
                    _merge_intervals(
                        [slice(s.start - start, s.stop - start) for s in curr],
                        merge_strategy,
                    ),
                )
            else:
                ans.append(sorted([slice(s.start - start, s.stop - start)
                           for s in curr], key=lambda x: (x.start, -x.stop)))
            start = end

        return ans

    def embed_bash(
        self,
        corpus: List[str],
        merge: bool = False,
        merge_strategy: str = "longest",
        skip_existing: bool = True,
        cache_dir: str = "cache",
        temp_file_prefix: str = "embed_temp",
    ) -> List[List[slice]]:  # pragma: no cover
        """
        Find duplicate byte slices using suffix array, with the origianl Google scripts.

        Parameters
        ----------
        corpus: List[str]
            List of documents.
        merge: bool
            Whether to merge overlapping intervals.
        merge_strategy: str
            Merge strategy.
        skip_existing: bool
            Whether to skip existing files.
        cache_dir: str
            Directory to store intermediate files.
        temp_file_prefix: str
            Prefix of temporary files.

        Returns
        -------
        List[List[slice]]
            List of duplicate byte slices.
        """
        assert merge_strategy in ["longest", "overlapping"], f"Invalid merge strategy: {merge_strategy}"
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        offsets = []
        start = 0
        with open(
            os.path.join(cache_dir, temp_file_prefix + f".{self.k}.txt"),
            "wb",
        ) as f:
            for doc in corpus:
                doc_bytes = doc.encode("utf-8")
                end = start + len(doc_bytes)
                offsets.append((start, end))
                start = end
                f.write(doc_bytes)

        logger.warning(
            "Make sure you have installed rust/cargo and initialized the submodule.",
        )

        def run_command(cmd: str):
            p = subprocess.Popen(
                cmd,
                shell=True,
                cwd=os.path.join(os.getcwd(), "deduplicate-text-datasets"),
            )
            code = p.wait()
            if code != 0:
                logger.error(f"Error running command: {cmd}")

        if not skip_existing or not os.path.exists(
            os.path.join(cache_dir, temp_file_prefix + f".{self.k}.byterange"),
        ):
            run_command(
                f"cargo run make --data-file {os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.txt')}",
            )
            run_command(
                f"python scripts/make_suffix_array.py {os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.txt')}",
            )
            run_command(
                f"cargo run self-similar --data-file {os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.txt')} --length-threshold {self.k} --cache-dir {cache_dir} --num-threads {os.cpu_count()}",
            )
            run_command(
                f"cargo run collect --data-file {os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.txt')} --length-threshold {self.k} --cache-dir {cache_dir} > {os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.byterange')}",
            )

        results: List[List[slice]] = [[] for _ in corpus]

        for idx, (x, y) in restore(
            offsets,
            os.path.join(
                cache_dir,
                temp_file_prefix + f".{self.k}.byterange",
            ),
        ):
            if y - x >= self.k:
                results[int(idx)].append(slice(x, y))

        if merge:
            for i in range(len(results)):
                results[i] = _merge_intervals(results[i], merge_strategy)

        return results

    def corss_embed_bash(
        self,
        corpus: List[str],
        query_corpus: List[str],
        merge: bool = False,
        merge_strategy: str = "longest",
        skip_existing: bool = True,
        cache_dir: str = "cache",
        temp_file_prefix: str = "embed_temp",
    ) -> Tuple[List[List[slice]], List[List[slice]]]:
        """
        Find duplicate byte slices for two datasets using suffix array, with the origianl Google scripts.

        Parameters
        ----------
        corpus: List[str]
            List of documents.
        query_corpus: List[str]
            List of query documents.
        merge: bool
            Whether to merge overlapping intervals.
        merge_strategy: str
            Merge strategy.
        skip_existing: bool
            Whether to skip existing files.
        cache_dir: str
            Directory to store intermediate files.
        temp_file_prefix: str
            Prefix of temporary files.

        Returns
        -------
        Tuple[List[List[slice]], List[List[slice]]]
            List of duplicate byte slices for corpus and query_corpus.
        """

        assert merge_strategy in ["longest", "overlapping"], f"Invalid merge strategy: {merge_strategy}"

        cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        corpus_a_path = os.path.join(cache_dir, temp_file_prefix + f".corpus_a.{self.k}.txt")
        corpus_b_path = os.path.join(cache_dir, temp_file_prefix + f".corpus_b.{self.k}.txt")
        corpus_a_output = os.path.join(cache_dir, temp_file_prefix + f".corpus_a.{self.k}.byterange")
        corpus_b_output = os.path.join(cache_dir, temp_file_prefix + f".corpus_b.{self.k}.byterange")

        offsets_a = []
        start = 0
        with open(
            corpus_a_path,
            "wb",
        ) as f:
            for doc in corpus:
                doc_bytes = doc.encode("utf-8")
                end = start + len(doc_bytes)
                offsets_a.append((start, end))
                start = end
                f.write(doc_bytes)

        offsets_b = []
        start = 0
        with open(
            corpus_b_path,
            "wb",
        ) as f:
            for doc in query_corpus:
                doc_bytes = doc.encode("utf-8")
                end = start + len(doc_bytes)
                offsets_b.append((start, end))
                start = end
                f.write(doc_bytes)

        logger.warning(
            "Make sure you have installed rust/cargo and initialized the submodule.",
        )

        def run_command(cmd: str):
            p = subprocess.Popen(
                cmd,
                shell=True,
                cwd=os.path.join(os.getcwd(), "deduplicate-text-datasets"),
            )
            code = p.wait()
            if code != 0:
                logger.error(f"Error running command: {cmd}")

        if not skip_existing or not os.path.exists(corpus_a_output) or not os.path.exists(corpus_b_output):
            run_command(
                f"cargo run make --data-file {corpus_a_path}",
            )
            run_command(
                f"cargo run make --data-file {corpus_b_path}",
            )
            run_command(
                f"python scripts/make_suffix_array.py {corpus_a_path}",
            )
            run_command(
                f"python scripts/make_suffix_array.py {corpus_b_path}",
            )
            run_command(
                f"cargo run across-similar --data-file-1 {corpus_a_path} --data-file-2 {corpus_b_path} --length-threshold {self.k} --cache-dir {cache_dir} --num-threads {os.cpu_count()}",
            )

            run_command(
                f"cargo run collect --data-file {corpus_a_path} --length-threshold {self.k} --cache-dir {cache_dir} > {corpus_a_output}",
            )
            run_command(
                f"cargo run collect --data-file {corpus_b_path} --length-threshold {self.k} --cache-dir {cache_dir} > {corpus_b_output}",
            )

        results_a: List[List[slice]] = [[] for _ in corpus]
        results_b: List[List[slice]] = [[] for _ in query_corpus]

        for idx, (x, y) in restore(
            offsets_a,
            corpus_a_output,
        ):
            if y - x >= self.k:
                results_a[int(idx)].append(slice(x, y))

        for idx, (x, y) in restore(
            offsets_b,
            corpus_b_output,
        ):
            if y - x >= self.k:
                results_b[int(idx)].append(slice(x, y))

        if merge:
            for i in range(len(results_a)):
                results_a[i] = _merge_intervals(results_a[i], merge_strategy)
            for i in range(len(results_b)):
                results_b[i] = _merge_intervals(results_b[i], merge_strategy)
        return results_a, results_b
