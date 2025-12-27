# pyright: reportAny=false
# pyright: reportExplicitAny=false
import math
import re
from collections.abc import Callable
from functools import partial
from itertools import permutations
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Self
from typing import override

import numpy as np
from bitarray import bitarray
from bitarray import frozenbitarray
from pydantic import model_validator

from text_dedup.config.algorithms.base import AlgorithmConfig
from text_dedup.utils.hashfunc import xxh3_hash
from text_dedup.utils.tokenization import ngrams


class Mask(NamedTuple):
    """
    A bit mask for data manipulation.

    For example, for a fingerprint data of length 12:
    - data: 010101111101
    - mask: 000001111000
    - mask_size: 4
    - start: 5
    - end: 9

    Parameters
    ----------
    mask: bitarray
        The bit mask.
    mask_size: int
        The size/length of the bit mask.
    start: int
        The start position of the bit mask.
    end: int
        The end position of the bit mask.
    """

    val: bitarray
    width: int
    start: int
    end: int

    def permute(self, x: bitarray, offset: int) -> bitarray:
        if offset > 0:
            return (x & self.val) << offset

        return (x & self.val) >> -offset

    def reverse(self, x: bitarray, offset: int) -> bitarray:
        if offset > 0:
            return (x & self.val) >> offset

        return (x & self.val) << -offset

    def reversed(self, offset: int) -> "Mask":
        if offset > 0:
            return Mask(val=self.val << offset, width=self.width, start=self.start, end=self.end)
        return Mask(val=self.val >> -offset, width=self.width, start=self.start, end=self.end)


class Permutation:
    def __init__(self, f: int, k: int, b: int, masks: list[Mask]) -> None:
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
            The block masks.
        """
        self.f: int = f
        self.k: int = k
        self.b: int = b

        width: int = 0
        self.widths: list[int] = []  # block widths
        self.offsets: list[int] = []  # block offsets
        self.reverse_masks: list[Mask] = []  # block reverse masks
        self.masks: list[Mask] = []  # block masks

        # Permutation illustrations:
        # data  : 010100
        # result: 000000
        # mask1 : 001100 -> 000000 | ((010100 & 001100) << 2)       -> 010000
        # mask2 : 000011 -> 010000 | ((010100 & 000011) << (4 - 2)) -> 010000
        # mask3 : 110000 -> 010000 | ((010100 & 110000) << (0 - 4)) -> 010001
        for mask in masks:
            offset = mask.start - width
            width += mask.width

            self.widths.append(mask.width)
            self.offsets.append(offset)
            self.masks.append(mask)
            self.reverse_masks.append(mask.reversed(offset))

        if sum(self.widths) != f:
            raise ValueError(f"The sum of block widths {sum(self.widths)} must be equal to the fingerprint size {f}")  # noqa: TRY003

        prefix_width = sum(self.widths[: b - k])
        self.search_mask: bitarray = bitarray(f)
        self.search_mask.setall(0)
        self.search_mask[:prefix_width] = 1
        self.search_mask = frozenbitarray(self.search_mask)

    def permute(self, x: bitarray) -> bitarray:
        """
        Permute the fingerprint.

        Parameters
        ----------
        x: bitarray
            The fingerprint to be permuted

        Returns
        -------
        bitarray
            The permuted fingerprint
        """
        result = bitarray(self.f)
        result.setall(0)
        for mask, offset in zip(self.masks, self.offsets, strict=True):
            result |= mask.permute(x, offset)
        return result

    def reverse(self, x: bitarray) -> bitarray:
        """
        Reverse the permutation.

        Parameters
        ----------
        x: bitarray
           The fingerprint to be reversed

        Returns
        -------
        bitarray
            The reversed fingerprint
        """
        result = bitarray(self.f)
        result.setall(0)
        for mask, offset in zip(self.reverse_masks, self.offsets, strict=True):
            result |= mask.reverse(x, offset)
        return result


def hamming_distance(a: bitarray, b: bitarray) -> int:
    """
    Compute the Hamming distance between two bitarrays.

    Parameters
    ----------
    a : bitarray
        The first bitarray.
    b : bitarray
        The second bitarray.

    Returns
    -------
    int
        The Hamming distance between the two bitarrays.

    Examples
    --------
    >>> _hamming_distance(bitarray("1010"), bitarray("1010"))
    0
    >>> _hamming_distance(bitarray("1010"), bitarray("0010"))
    1
    """
    return (a ^ b).count(1)


def _unsigned_hash(obj: bytes, hash_func: Callable[[bytes], int], length: int) -> bitarray:
    """
    Compute a hash of an object.

    It doesn't really matter what hash function to use, as long as it is consistent.

    Parameters
    ----------
    obj: bytes
        The object to hash.
    hash_func: Callable
        The hash function to use.

    Returns
    -------
    bitarray
        The hash of the object.

    Examples
    --------
    >>> len(_unsigned_hash(b"hello world", xxh3_64_digest))
    64
    >>> len(_unsigned_hash(b"hello world", xxh3_128_digest))
    128
    """
    result = bitarray(0)
    result.frombytes(hash_func(obj).to_bytes(length=length))
    return result


def compute(hashes: list[bitarray]) -> bitarray:
    """
    Compute the Simhash of a list of hashes.

    Notes to myself: You tried porting this to Cython, but it didn't improve the performance.
    Others have experimented with numpy types and operators, but it didn't improve performance

    Parameters
    ----------
    hashes : List[int]
        The list of hashes.

    Returns
    -------
    bitarray
        The Simhash of the list of hashes.

    Examples
    --------
    >>> from bitarray.util import int2ba, ba2int
    >>> res = compute([int2ba(13352372148217134600, length=64), int2ba(5020219685658847592, length=64)])
    >>> ba2int(res)
    74633958390507528
    """
    if not hashes:
        raise ValueError("Cannot compute simhash from empty hash list")  # noqa: TRY003

    sigs = np.asarray([h.tolist() for h in hashes], dtype=int)
    sig = np.where(np.sum(2 * sigs - 1, axis=0) > 0, True, False)
    res = bitarray()
    res.extend(sig.tolist())
    return res


class SimHashAlgorithmConfig(AlgorithmConfig):
    algorithm_name: Literal["simhash"] = "simhash"  # pyright: ignore[reportIncompatibleVariableOverride]
    ngram_size: int = 3
    f: int = 64
    bit_diff: int = 3
    num_bucket: int = 4
    min_length: int = 5
    check_false_positive: bool = False
    jaccard_threshold: float = 0.5

    _max_block_size: int
    _min_block_size: int
    _x: int
    _y: int
    _perms: list[Permutation]

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.num_bucket <= self.bit_diff:
            raise ValueError("num_bucket must be greater than bit_diff")  # noqa: TRY003
        return self

    @override
    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._max_block_size = math.ceil(self.f / self.num_bucket)
        self._min_block_size = math.floor(self.f / self.num_bucket)

        # solve: max_block_size * x + min_block_size * y == f
        x, y = 0, 0
        while True:
            x += 1
            if (self.f - x * self._max_block_size) % self._min_block_size == 0:
                y = (self.f - x * self._max_block_size) // self._min_block_size
                break

        assert x * self._max_block_size + y * self._min_block_size == self.f, (  # noqa: S101
            f"{x=} w/ {self._max_block_size}, {y=} w/ {self._min_block_size} are invalid"
        )
        self._x = x
        self._y = y
        self._perms = self.create_permutations()

    @property
    def hash_func(self) -> Callable[[bytes], int]:
        match self.f:
            case 128:
                return partial(xxh3_hash, seed=self.seed, bits=128)
            case 64:
                return partial(xxh3_hash, seed=self.seed, bits=64)
            case _:
                return partial(xxh3_hash, seed=self.seed, bits=self.f)

    def _get_tokenize_func(self) -> Callable[[str], list[str]]:
        NON_ALPHA = re.compile(r"\W", re.UNICODE)

        def f(content: str) -> list[str]:
            return [t for t in NON_ALPHA.split(content.lower()) if t]

        return f

    def get_ngrams_func(self) -> Callable[[str], set[bytes]]:
        tokenize_func = self._get_tokenize_func()

        def f(content: str) -> set[bytes]:
            return {
                bytes(" ".join(t).lower(), "utf-8")
                for t in ngrams(tokenize_func(content), self.ngram_size, self.min_length)
            }

        return f

    def get_embed_func(self) -> Callable[[list[str], list[int]], dict[str, list[int | bytes]]]:
        tokenizer = self.get_ngrams_func()

        def f(text_col: list[str], idx_col: list[int]) -> dict[str, Any]:
            """
            Calculate the simhash signature of a text.

            Parameters
            ----------
            content : str
                The text to be hashed.
            idx : int
                The index of the text.
            ngram : int
                The ngram size.
            hash_func : Callable
                hash function to use

            Returns
            -------
            Dict[str, Any]
                The simhash signature and the index of the text as a dictionary.

            Examples
            --------
            >>> res = embed_func("hello world", 0, ngram=3, permutations=None, hash_func=xxh3_64_digest)
            >>> res[INDEX_COLUMN]
            0
            >>> len(res[SIGNATURE_COLUMN])
            8
            """

            tokens = tokenizer(text_col[0])
            if tokens:
                sig = compute([_unsigned_hash(t, self.hash_func, length=self.f // 8) for t in tokens])
            else:
                sig = bitarray(self.f)
                sig.setall(0)
            keys: list[tuple[bytes, bytes]] = []
            if self._perms:
                for permutation in self._perms:
                    keys.append((
                        permutation.search_mask.tobytes(),
                        (permutation.permute(sig) & permutation.search_mask).tobytes(),
                    ))
            val = sig.tobytes()
            return {
                "__key__": keys,
                "__val__": [val for _ in keys],
                self.internal_index_column: [idx_col[0] for _ in keys],
            }

        return f

    @staticmethod
    def hamming_distance(a: bitarray, b: bitarray) -> int:
        """
        Compute the Hamming distance between two bitarrays.

        Parameters
        ----------
        a : bitarray
            The first bitarray.
        b : bitarray
            The second bitarray.

        Returns
        -------
        int
            The Hamming distance between the two bitarrays.

        Examples
        --------
        >>> _hamming_distance(bitarray("1010"), bitarray("1010"))
        0
        >>> _hamming_distance(bitarray("1010"), bitarray("0010"))
        1
        """
        return (a ^ b).count(1)

    def create_permutations(self) -> list[Permutation]:
        """
        Create permutations for f bits and b blocks with k-bit difference allowed.

        Returns
        -------
        List[Permutation]
            The permutations

        Examples
        --------
        >>> from bitarray.util import urandom
        >>> perms = _create_permutations(128, 3, 4)
        >>> len(perms)
        4
        >>> data = urandom(128)
        >>> for perm in perms:
        ...     assert perm.reverse(perm.permute(data)) == data, f"{perm.reverse(perm.permute(data))} != {data}"
        """

        masks: list[Mask] = []
        start = end = 0

        for _ in range(self.num_bucket):
            block_size = self._max_block_size if self._x > 0 else self._min_block_size
            start, end = end, min(end + block_size, self.f)
            if start >= end:
                break
            mask: bitarray = bitarray(self.f)
            mask.setall(0)
            mask[start:end] = 1
            masks.append(
                Mask(
                    val=mask,
                    width=end - start,
                    start=start,
                    end=end,
                )
            )

        results: list[Permutation] = []

        # b - k many blocks must be the same to only allow k-bit difference
        indices = set(range(len(masks)))
        for fixed in permutations(indices, self.num_bucket - self.bit_diff):
            changing = sorted(indices - set(fixed))
            blocks = [masks[i] for i in fixed] + [masks[i] for i in changing]
            results.append(Permutation(self.f, self.bit_diff, self.num_bucket, blocks))

        return results
