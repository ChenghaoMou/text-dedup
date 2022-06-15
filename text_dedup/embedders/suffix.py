#!/usr/bin/env python
# @Date    : 2022-04-02 12:15:08
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import logging
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List

from tqdm import tqdm

from text_dedup.utils.sa import construct_sa
from text_dedup.utils.suffix_restore import restore

logger = logging.getLogger('text-dedup')


def _merge_intervals(
    slices: List[slice],
    merge_strategy: str = 'overlapping',
) -> List[slice]:
    """Merge overlapping intervals.

    Parameters
    ----------
    slices : List[slice]
        List of intervals

    Returns
    -------
    List[slice]
        List of merged intervals

    Examples
    --------
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

        if merge_strategy == 'overlapping':
            if prev.stop >= current.start:
                merged[-1] = slice(prev.start, max(prev.stop, current.stop))
            else:
                merged.append(current)
        elif merge_strategy == 'longest':
            if current.stop <= prev.stop:  # ignore substrings
                continue
            else:
                merged.append(current)

    return merged


@dataclass
class SuffixArrayEmbedder:

    """
    Find duplicate byte slices using suffix array.

    Parameters
    ----------
    k: int
        Minimum length of the byte slices.
    """

    k: int = 100

    def embed(
        self,
        corpus: List[str],
        merge: bool = False,
        merge_strategy: str = 'longest',
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

        assert merge_strategy in ['longest', 'overlapping']

        string = ''.join(corpus)
        lengths = [len(s) for s in corpus]

        sa = construct_sa(string)
        slices = []

        for x, y in tqdm(
            zip(sa[:-1], sa[1:]),
            total=len(sa) - 1,
            desc='Suffix array querying',
        ):
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
                    curr.append(slice(s.start, end))
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
                ans.append([slice(s.start - start, s.stop - start) for s in curr])
            start += length

        return ans

    def embed_bash(
        self,
        corpus: List[str],
        skip_existing: bool = True,
        cache_dir: str = 'cache',
        temp_file_prefix: str = 'embed_temp',
    ) -> List[List[slice]]:  # pragma: no cover
        """
        Find duplicate byte slices using suffix array, with the origianl Google scripts.

        Parameters
        ----------
        corpus: List[str]
            List of documents.
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

        cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        offsets = []
        start = 0
        with open(
            os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.txt'),
            'wb',
        ) as f:
            for doc in corpus:
                doc_bytes = doc.encode('utf-8')
                end = start + len(doc_bytes)
                offsets.append((start, end))
                start = end
                f.write(doc_bytes)

        logger.warning(
            'Make sure you have installed rust/cargo and initialized the submodule.',
        )

        def run_command(cmd: str):
            p = subprocess.Popen(
                cmd,
                shell=True,
                cwd=os.path.join(os.getcwd(), 'deduplicate-text-datasets'),
            )
            code = p.wait()
            if code != 0:
                logger.error(f'Error running command: {cmd}')

        if not skip_existing or not os.path.exists(
            os.path.join(cache_dir, temp_file_prefix + f'.{self.k}.byterange'),
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
                temp_file_prefix + f'.{self.k}.byterange',
            ),
        ):
            if y - x >= self.k:
                results[int(idx)].append(slice(x, y))
        return results
