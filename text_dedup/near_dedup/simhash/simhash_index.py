# Created by 1e0n in 2013
# modified by Chenghao Mou in 2022
import logging
import math
from itertools import permutations
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

from tqdm import tqdm

from text_dedup.base import Fingerprint
from text_dedup.utils import StorageDict
from text_dedup.utils import create_storage

logger = logging.getLogger("text_dedup")


def _hamming_distance(a: int, b: int) -> int:
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
    >>> _hamming_distance(0b1010, 0b0101)
    4
    >>> _hamming_distance(0b1010, 0b1010)
    0
    """

    # c = a ^ b
    # ans = 0
    # while c:
    #     ans += 1
    #     c &= c - 1
    # return ans

    return (a ^ b).bit_count()


class Permutation:
    def __init__(self, f: int, k: int, b: int, masks: List[Tuple[int, int, int, int]]) -> None:
        """
        A permutation object for bit manipulation.

        More details about this permutation can be found in https://github.com/seomoz/simhash-py#architecture.

        Parameters
        ----------
        f: int
            The fingerprint bit length
        k: int
            The bit difference allowed
        b: int
            The number of blocks
        masks:
            The block masks generated from `_create_permutations`
        """
        self.f = f
        self.k = k
        self.b = b

        width: int = 0
        self.widths: List[int] = []  # block widths
        self.offsets: List[int] = []  # block offsets
        self.reverse_masks: List[int] = []  # block reverse masks
        self.masks: List[int] = []  # block masks
        for mask, mask_size, start, _ in masks:
            self.widths.append(mask_size)
            width += mask_size
            offset = f - width - start
            self.offsets.append(offset)
            if offset > 0:
                self.reverse_masks.append(mask << offset)
            else:
                self.reverse_masks.append(mask >> -offset)

            self.masks.append(mask)

        prefix_width = sum(self.widths[: b - k])
        self.search_mask: int = 0
        for i in range(f):
            if i < prefix_width:
                self.search_mask += 1
            self.search_mask <<= 1

    def permute(self, x: int) -> int:
        """
        Permute the fingerprint.

        Parameters
        ----------
        x: int
            The fingerprint to be permuted

        Returns
        -------
        int
            The permuted fingerprint
        """
        result = 0

        for mask, offset in zip(self.masks, self.offsets):
            if offset > 0:
                result |= (x & mask) << offset
            else:
                result |= (x & mask) >> -offset

        return result

    def reverse(self, x: int) -> int:
        """
        Reverse the permutation.

        Parameters
        ----------
        x: int
           The fingerprint to be reversed

        Returns
        -------
        int
            The reversed fingerprint
        """
        result = 0
        for mask, offset in zip(self.reverse_masks, self.offsets):
            if offset > 0:
                result |= (x & mask) >> offset
            else:
                result |= (x & mask) << -offset
        return result


def _create_permutations(f: int, k: int, b: int) -> List[Permutation]:
    """
    Create permutations for f bit, b blocks, and k bit difference allowed.

    Parameters
    ----------
    f: int
        The fingerprint to be permuted
    k: int
        The bit difference allowed
    b: int
        The number of blocks

    Returns
    -------
    List[Permutation]
        The permutations
    """
    block_size: int = math.ceil(f / b)
    masks: List[Tuple[int, int, int, int]] = []
    for i in range(b):
        mask: int = 0
        start, end = i * block_size, min((i + 1) * block_size, f)
        for j in range(start, end):
            mask |= 1 << j
        masks.append(
            (
                mask,
                end - start,
                start,
                end,
            )
        )

    results: List[Permutation] = []
    for leading_blocks in permutations(masks, b - k):
        blocks = list(leading_blocks)
        for record in masks:
            if record not in blocks:
                blocks.append(record)
        results.append(Permutation(f, k, b, blocks))

    return results


class SimHashIndex(object):
    def __init__(
            self,
            fingerprints: List[Tuple[int, Fingerprint]],
            f: int = 64,
            k: int = 3,
            b: int = 4,
            storage_config: Optional[Dict[str, Any]] = None,
            verbose: bool = False,
    ):
        assert b > k, "b must be greater than k"

        self.k = k
        self.b = b
        self.f = f

        self.bucket: StorageDict = create_storage(storage_config)
        self.permutations = _create_permutations(f, k, b)

        if len(self.bucket) == 0:
            for idx, fingerprint in tqdm(fingerprints, desc="Indexing...", disable=not verbose):
                self.add(idx, fingerprint)

        logger.info(
            f"""Simhash index created with {len(fingerprints)} signatures, {len(self.bucket)} buckets.""")

        if verbose:
            largest_bucket: Any = max(self.bucket, key=lambda x: len(self.bucket[x]))
            logger.info(
                f"Maximum bucket size: {len(self.bucket[largest_bucket])} with the key {largest_bucket}")

    def get_near_duplicates(self, fingerprint: int) -> List[Any]:
        ans = set()
        for key in self.get_keys(fingerprint):
            for idx, other_fingerprint in self.bucket[key]:
                if _hamming_distance(fingerprint, other_fingerprint) <= self.k:
                    ans.add(idx)
        return list(ans)

    def add(self, idx: int, fingerprint: int):
        for key in self.get_keys(fingerprint):
            self.bucket.add(key, (idx, fingerprint))

    def get_keys(self, fingerprint: int) -> Generator[Tuple[int, int], None, None]:
        for permutation in self.permutations:
            yield permutation.search_mask, permutation.permute(fingerprint) & permutation.search_mask
