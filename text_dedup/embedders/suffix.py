#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:15:08
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from collections import deque
from dataclasses import dataclass
from typing import List

from text_dedup.embedders import Embedder
from text_dedup.utils.sa import construct_sa


def _merge_intervals(slices: List[slice], merge_strategy: str = 'overlapping') -> List[slice]:
    """Merge overlapping intervals.

    Parameters
    ----------
    slices : List[slice]
        List of intervals

    Returns
    -------
    List[slice]
        List of merged intervals
    """
    if len(slices) == 0:
        return []

    slices = sorted(list(map(lambda s: slice(s[0], s[1]), set([(s.start, s.stop) for s in slices]))), key=lambda x: (x.start, - x.stop))

    merged = []
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
            if current.stop <= prev.stop: # ignore substrings
                continue
            else:
                merged.append(current)

    return merged

@dataclass
class SuffixArrayEmbedder(Embedder):

    k: int = 50

    def embed(self, corpus: List[str], merge: bool = False, merge_strategy: str = 'longest') -> List[List[slice]]:

        assert merge_strategy in ['longest', 'overlapping']

        string = ''.join(corpus)
        lengths = [len(s) for s in corpus]

        sa = construct_sa(string)
        slices = []

        for x, y in zip(sa[:-1], sa[1:]):
            matched_length = 0
            while x + matched_length < len(string) and y + matched_length < len(string) and string[x + matched_length] == string[y + matched_length]:
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
                ans.append(_merge_intervals([slice(s.start - start, s.stop - start) for s in curr], merge_strategy))
            else:
                ans.append([slice(s.start - start, s.stop - start) for s in curr])
            start += length
        
        return ans
            



