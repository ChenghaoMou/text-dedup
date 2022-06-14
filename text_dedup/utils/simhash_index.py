# Created by 1e0n in 2013
# modified by Chenghao Mou in 2022
import collections
from typing import Any, Dict, Generator, List, Set, Tuple


def hamming_distance(a: int, b: int) -> int:
    """
    Compute the Hamming distance between two integers.

    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.

    Returns
    -------
    int
        The Hamming distance between the two integers.

    Examples
    --------
    >>> hamming_distance(0b1010, 0b0101)
    4
    >>> hamming_distance(0b1010, 0b1010)
    0
    """

    c = a ^ b
    ans = 0
    while c:
        ans += 1
        c &= c - 1
    return ans


class SimhashIndex(object):

    def __init__(
        self,
        fingerprints: List[Tuple[int, int]],
        f: int = 64,
        k: int = 3,
        b: int = 4,
    ):
        assert b > k, 'b must be greater than k'

        self.k = k
        self.b = b
        self.f = f
        self.bucket: Dict[Tuple[int, int], Set[Tuple[int, int]]
                          ] = collections.defaultdict(set)

        if len(self.bucket) == 0:
            for idx, fingerprint in fingerprints:
                self.add(idx, fingerprint)

    def get_near_dups(self, fingerprint: int) -> List[Any]:
        ans = set()
        for key in self.get_keys(fingerprint):
            for idx, other_fingerprint in self.bucket[key]:
                if hamming_distance(fingerprint, other_fingerprint) <= self.k:
                    ans.add(idx)
        return list(ans)

    def add(self, idx: int, fingerprint: int):
        for key in self.get_keys(fingerprint):
            self.bucket[key].add((idx, fingerprint))

    @property
    def offsets(self):
        return [self.f // self.b * i for i in range(self.b)]

    def get_keys(self, fingerprint: int) -> Generator[Tuple[int, int], None, None]:
        offsets = self.offsets + [self.f]
        for offset_idx, offset in enumerate(self.offsets):
            mask: int = 2 ** (offsets[offset_idx + 1] - offset) - 1
            mask_value: int = fingerprint >> offset & mask
            yield (offset_idx, mask_value)
