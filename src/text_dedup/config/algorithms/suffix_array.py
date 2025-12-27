# pyright: reportAny=false
# pyright: reportExplicitAny=false
import subprocess
from collections import deque
from collections.abc import Generator
from pathlib import Path
from pathlib import PosixPath
from typing import Literal

from text_dedup.config.algorithms.base import AlgorithmConfig


class SuffixArrayAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["suffix_array"] = "suffix_array"
    merge_strategy: Literal["longest", "overlapping"] = "longest"
    length_threshold: int = 100
    google_repo_path: str = "third_party/deduplicate-text-datasets"
    cache_dir: str = ".cache"

    def merge_intervals(
        self,
        intervals: list[slice],
        merge_strategy: Literal["longest", "overlapping"] = "longest",
    ) -> list[slice]:
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
        >>> merge_intervals(
        ...     [
        ...         slice(0, 10, None),
        ...         slice(1, 11, None),
        ...         slice(2, 12, None),
        ...         slice(3, 13, None),
        ...         slice(4, 14, None),
        ...         slice(5, 15, None),
        ...         slice(6, 16, None),
        ...         slice(7, 21, None),
        ...     ],
        ...     merge_strategy="overlapping",
        ... )
        [slice(0, 21, None)]
        >>> merge_intervals(
        ...     [
        ...         slice(0, 10, None),
        ...         slice(1, 11, None),
        ...         slice(2, 12, None),
        ...         slice(3, 13, None),
        ...         slice(4, 14, None),
        ...         slice(5, 15, None),
        ...         slice(6, 16, None),
        ...         slice(7, 21, None),
        ...     ],
        ...     merge_strategy="longest",
        ... )  # doctest: +ELLIPSIS
        [slice(0, 10, None), slice(1, 11, None), slice(2, 12, None), ... slice(7, 21, None)]
        >>> merge_intervals([slice(0, 2), slice(2, 4), slice(4, 5)], "overlapping")
        [slice(0, 5, None)]
        >>> merge_intervals([slice(0, 4), slice(2, 4), slice(4, 5)], "longest")
        [slice(0, 4, None), slice(4, 5, None)]
        >>> merge_intervals(
        ...     [slice(0, 10, None), slice(0, 10, None), slice(0, 10, None), slice(0, 10, None), slice(0, 10, None)]
        ... )
        [slice(0, 10, None)]
        """
        if len(intervals) == 0:
            return []

        q = deque(
            sorted(
                (slice(s[0], s[1]) for s in {(s.start, s.stop) for s in intervals}),
                key=lambda x: (x.start, -x.stop),
            )
        )

        merged: list[slice] = []

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

    def restore(  # noqa: C901
        self,
        boundaries: list[slice],
        segments: str | Path | list[slice],
    ) -> Generator[tuple[int, slice], None, None]:
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
        >>> list(
        ...     restore(
        ...         [slice(0, 10, None), slice(10, 20, None)],
        ...         [slice(0, 5, None), slice(5, 10, None), slice(5, 15, None), slice(5, 19, None)],
        ...     )
        ... )
        [(0, slice(0, 5, None)), (0, slice(5, 10, None)), (1, slice(0, 5, None)), (1, slice(0, 9, None))]
        """
        indices: deque[slice] = deque([])

        if isinstance(segments, str | PosixPath):
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
        self,
        boundaries: list[slice],
        segments: str | Path | list[slice],
        k: int,
        merge_strategy: Literal["longest", "overlapping"] = "longest",
    ) -> tuple[list[list[slice]], int]:
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
        ...     [slice(0, 10, None), slice(10, 20, None)],
        ...     [slice(0, 5, None), slice(5, 10, None), slice(12, 19, None)],
        ...     5,
        ...     "longest",
        ... )
        ([[slice(0, 5, None), slice(5, 10, None)], [slice(2, 9, None)]], 17)
        >>> restore_and_merge(
        ...     [slice(0, 10, None), slice(10, 20, None)],
        ...     [slice(0, 5, None), slice(5, 10, None), slice(12, 19, None)],
        ...     5,
        ...     "overlapping",
        ... )
        ([[slice(0, 10, None)], [slice(2, 9, None)]], 17)
        """
        duplicate_size = 0
        results: list[list[slice]] = [[] for _ in boundaries]
        for idx, s in self.restore(boundaries, segments):
            if s.stop - s.start >= k:
                results[int(idx)].append(s)
        for i in range(len(results)):
            results[i] = self.merge_intervals(results[i], merge_strategy)
            duplicate_size += sum([s.stop - s.start for s in results[i]])
        return results, duplicate_size

    def run_command(self, cmd: str, cwd: str) -> None:
        p = subprocess.Popen(  # noqa: S602
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            error_msg = f"Command {cmd} failed with code {p.returncode}. CWD: {cwd}"
            if stdout:
                error_msg += f"\nstdout:\n{stdout.decode(errors='replace')}"
            if stderr:
                error_msg += f"\nstderr:\n{stderr.decode(errors='replace')}"
            raise RuntimeError(error_msg)

    def clean_up(self, text: str, slices: list[slice]) -> str:
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
