#!/usr/bin/env python


########################################################################
#                   FastGeneralizedSuffixArrays v0.1                   #
#                       (c) 2011 Mark Mazumder                         #
#                             markmaz.com                              #
########################################################################

from typing import List
from typing import Sequence

from tqdm import tqdm

from text_dedup.base import Fingerprint
from text_dedup.exact_dedup.suffix_array.base import SuffixArrayDeduplicator
from text_dedup.exact_dedup.suffix_array.utils import restore_and_merge


class Triple(object):
    """Represent each sortable character in R with three integers"""

    # todo: input validation, errors
    def __init__(self, T, idx, length):
        def t_i(i):
            return T[i] if i < length else 0

        self._triple = (t_i(idx), t_i(idx + 1), t_i(idx + 2))
        self._index = idx
        self._rank = None
        self._rpos = None

    @property
    def triple(self):
        """Character for R_k strings"""
        return self._triple

    @property
    def index(self):
        """Position of R_k character in source string"""
        return self._index

    @property
    def rpos(self):
        """Sorted order of R_k charcter"""
        return self._rpos

    @rpos.setter
    def rpos(self, pos):
        self._rpos = pos

    @property
    def rank(self):
        """Sorted order of R_k charcter"""
        return self._rank

    @rank.setter
    def rank(self, pos):
        self._rank = pos

    def __repr__(self):
        return 'Triple({0}, {1}, {2})'.format(self.triple, self.index, self.rank)


class NonsamplePair(object):
    # todo: property decorators for validation
    def __init__(self, T, idx, S_i_ranks):
        self.index = idx
        self.pair = None
        max_index = len(T)
        if idx < max_index:
            self.pair = (T[self.index], S_i_ranks[self.index + 1])
        else:
            self.pair = (
                0,
                S_i_ranks[self.index + 1],
            )  # defined to be 0 by KS algorithm


# Recursive Karkkainen-Sanders implementation
#   Input: list of integers (representing characters)
#   Returns suffix array for list
def ksa(T):
    length = len(T)  # n
    # B_k = { i \in [0,n] | i mod 3 = k }
    B_0, B_1, B_2 = (
        range(0, length + 1, 3),
        range(1, length + 1, 3),
        range(2, length + 1, 3),
    )

    # karkkainen-sanders step 1: sort sample suffixes
    # R_0 = [Triple(T, idx, length) for idx in B_0]
    R_1 = [Triple(T, idx, length) for idx in B_1]
    R_2 = [Triple(T, idx, length) for idx in B_2]

    R = R_1 + R_2
    # enable reverse-lookup of characters in R from a list of sorted characters from R
    for i, r_char in enumerate(R):
        r_char.rpos = i
    sorted_suffixes_R = sorted(R, key=lambda suffix_char: suffix_char.triple)

    # Enables 0 as unique terminating character by starting ranks at 1
    def rank_suffixes(suffixes, rank=1):
        for i, suffix in enumerate(suffixes):
            if i > 0 and suffix.triple != suffixes[i - 1].triple:
                rank += 1
            suffix.rank = rank
        return rank

    rank = rank_suffixes(sorted_suffixes_R)
    R_prime = [suffix.rank for suffix in R]

    # recursive call
    if rank < len(R):  # we had repeats of characters of R, make a recursive call to sort
        R_prime_suffix_array = ksa(R_prime)
    else:
        # directly form suffix array
        R_prime_suffix_array = [len(R)] + [suffix.rpos for suffix in sorted_suffixes_R]
    # why plus 3? -> additiionally define rank(S_(n+1) = rank(S_(n+2)) = 0
    rank_Si = [None] * (length + 3)
    rank_Si[-2] = rank_Si[-1] = 0

    # build rank(S_i) lookup array
    for i, SAi in enumerate(R_prime_suffix_array):
        if SAi < len(R):  # ignore the index pointing to the terminating character of R_prime
            rank_Si[R[SAi].index] = i

    sorted_suffixes_R = [R[i] for i in R_prime_suffix_array[1:]]

    # karkkainen-sanders step 2: sort nonsample suffixes
    nonsample_suffix_pairs = [NonsamplePair(T, idx, rank_Si) for idx in B_0]
    sorted_nonsample_suffix_pairs = sorted(nonsample_suffix_pairs, key=lambda p: p.pair)

    # karkkainen-sanders step 3: merge
    cur_Sc, cur_Sb0 = 0, 0
    objs_SA = []

    def getT(idx):
        if idx < len(T):
            return T[idx]
        return 0

    while cur_Sc < len(sorted_suffixes_R) and cur_Sb0 < len(sorted_nonsample_suffix_pairs):
        i = sorted_suffixes_R[cur_Sc].index
        j = sorted_nonsample_suffix_pairs[cur_Sb0].index
        if i % 3 == 1:  # i in B_1
            # S_i =< S_j iff (T[i], rank(S_t+1) =< (t_j, rank(s_j+1))
            if (getT(i), rank_Si[i + 1]) < (getT(j), rank_Si[j + 1]):
                objs_SA.append(sorted_suffixes_R[cur_Sc])
                cur_Sc += 1
            else:
                objs_SA.append(sorted_nonsample_suffix_pairs[cur_Sb0])
                cur_Sb0 += 1
        else:  # i in B_2
            if (getT(i), getT(i + 1), rank_Si[i + 2]) < (
                    getT(j),
                    getT(j + 1),
                    rank_Si[j + 2],
            ):
                objs_SA.append(sorted_suffixes_R[cur_Sc])
                cur_Sc += 1
            else:
                objs_SA.append(sorted_nonsample_suffix_pairs[cur_Sb0])
                cur_Sb0 += 1

    objs_SA += sorted_suffixes_R[cur_Sc:]
    objs_SA += sorted_nonsample_suffix_pairs[cur_Sb0:]
    SA = [suffix_object.index for suffix_object in objs_SA]
    return SA


# Python Implementation of SuffixArrayDeduplicator
class PythonSuffixArrayDeduplicator(SuffixArrayDeduplicator):

    def fit(self, data: Sequence[str]):
        raise NotImplementedError(
            "fit is not implemented for PythonSuffixArrayDeduplicator")

    def predict(self, data: Sequence[str]) -> List[List[slice]]:
        raise NotImplementedError(
            "predict is not implemented for PythonSuffixArrayDeduplicator")

    def fit_predict(self, data: Sequence[str]) -> List[List[slice]]:
        """
        Fit and predict in one step.

        Parameters
        ----------
        data: Sequence[str]
            A sequence of strings to be fingerprinted.

        Returns
        -------
        List[List[slice]]
            A list of lists of slices, where each slice is a duplicate.

        Examples
        --------
        >>> PythonSuffixArrayDeduplicator(k=1).fit_predict(["abc", "abc", "abc"])
        [[slice(0, 3, None)], [slice(0, 3, None)], [slice(0, 3, None)]]
        """
        encoded_data: List[bytes] = [d.encode('utf-8') for d in data]
        text: bytes = b''.join(encoded_data)
        sequence: List[int] = [b for b in text]
        sa = ksa(sequence)
        n: int = len(text)

        boundaries: List[slice] = []
        start = 0
        for d in encoded_data:
            length = len(d)
            boundaries.append(slice(start, start + length))
            start += length

        slices = []
        for x, y in tqdm(
                zip(sa[:-1], sa[1:]),
                total=len(sa) - 1,
                desc="Querying...",
        ):
            matched_length: int = 0
            while (
                    x + matched_length < n
                    and y + matched_length < n
                    and text[x + matched_length] == text[y + matched_length]
            ):
                matched_length += 1

            if matched_length >= self.k:
                slices.append(slice(x, x + matched_length))
                slices.append(slice(y, y + matched_length))

        segments = sorted(map(lambda x: slice(x[0], x[1]), {
                          (s.start, s.stop) for s in slices}), key=lambda s: (s.start, - s.stop))
        return restore_and_merge(boundaries, segments, self.k, self.merge_strategy)
