#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-07-24 09:19:36
# @Author  : Chenghao Mou (mouchneghao@gmail.com)

"""Deduplicating Training Data Makes Language Models Better."""

from typing import List, Any, Tuple
from multiprocessing import Manager
from ctypes import c_char_p
import multiprocessing as mp
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


def similar(x: int, y: int, S: Any, k: int) -> bool:
    """Whether S[x:x+k] is the same as S[y:y+k].

    Parameters
    ----------
    x : int
        [description]
    y : int
        [description]
    S : Any
        [description]
    k : int
        [description]

    Returns
    -------
    bool
        [description]
    """
    if x == y:
        return True

    return (
        x + k <= len(S.value)
        and y + k <= len(S.value)
        and S.value[x : x + k] == S.value[y : y + k]
    )


def group(x: str, patterns: str) -> List[int]:
    """Find patterns that are present in string x.

    Parameters
    ----------
    x : str
        A document string
    patterns : str
        Patterns to search for

    Returns
    -------
    List[int]
        List of indices of which patterns are present in string x
    """
    result = []
    for idx, pattern in enumerate(patterns):
        if pattern in x:
            result.append(idx)
    return result


class SuffixArray:
    def __init__(self, k: int = 50):
        self.array = []
        self.k = k

    def fit_transform(self, data: List[str]) -> Tuple[List[str], np.ndarray]:
        """Find duplicate substrings in the data.

        Parameters
        ----------
        data : List[str]
            List of documents.

        Returns
        -------
        Tuple[List[str], np.ndarray]
            List of duplicate substrings and a matrix where each row is a document and each column is a substring.

        Examples
        --------
        >>> array = SuffixArray(k = 9)
        >>> duplicates, groups = array.fit_transform(["This is a sentence.", "This is another sentences.", "This is a question.", "hello world"] * 10)
        >>> assert len(duplicates) == groups.shape[1], "Invalid number of columns"
        >>> assert groups.shape[0] == 40, "Invalid number of rows"
        """
        S = "".join(data)
        suffixes = []
        for i in range(len(S)):
            suffixes.append(S[i:])

        self.array = np.argsort(suffixes)

        # Find duplicated substrings
        manager = Manager()
        shared = manager.Value(c_char_p, S)

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(
                similar,
                [(x, y, shared, self.k) for x, y in sliding_window_view(self.array, 2)],
            )

        duplicates = []
        for idx, dup in zip(self.array, results):
            if dup:
                duplicates.append(S[idx : idx + self.k])

        # Find duplicated documents
        try:
            from multiprocessing import shared_memory

            shared = shared_memory.ShareableList(duplicates)
        except ImportError as e:
            print(
                f"The following error was: \n{e}\n\n"
                + "This was likely raised since you are not running python 3.8 or higher."
                + " Continuing without a shared memory file which is likely be inefficient."
            )
            shared = duplicates
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(group, [(d, shared) for d in data])

        shared.shm.close()
        shared.shm.unlink()
        del shared

        groups = np.zeros((len(data), len(duplicates)), dtype=bool)
        for i, x in enumerate(results):
            for y in x:
                groups[i, y] = 1

        return duplicates, groups
