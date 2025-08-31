# pyright: reportAny=false
# pyright: reportExplicitAny=false
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import override

import numpy as np
import numpy.typing as npt
import regex as re
from scipy.integrate import quad as integrate

from text_dedup.config.algorithms.base import AlgorithmConfig
from text_dedup.utils.tokenization import ngrams


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> tuple[int, int]:  # pragma: no cover
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of false positive and false negative areas, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.
    """

    def false_positive_area(threshold: float, b: int, r: int) -> float:
        """Source: `datasketch.lsh`"""

        def proba(s: float) -> float:
            return float(1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, 0.0, threshold)
        return float(a)

    def false_negative_area(threshold: float, b: int, r: int) -> float:
        """Source: `datasketch.lsh`"""

        def proba(s: float) -> float:
            return float(1 - (1 - (1 - s ** float(r)) ** float(b)))

        a, _ = integrate(proba, threshold, 1.0)
        return float(a)

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["minhash"] = "minhash"
    hash_bits: int = 64
    hash_func_name: Literal["sha1", "xxh3"] = "sha1"
    bands: int | None = None
    rows: int | None = None
    num_perm: int
    ngram_size: int = 1
    threshold: float
    min_length: int = 5
    false_positive_weight: float = 0.5
    false_negative_weight: float = 0.5
    check_false_positive: bool = False
    _modulo_prime: np.uint
    _max_hash: np.uint
    _dtype: type

    # 64bit config is backwards compatibility mode.
    # it uses 64bit types but almost entirely 32bit data, except for one mersenne prime 2^61
    # why legacy implementations used mersenne primes for modulo:
    # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
    # (dtype, max_hash, modulo_prime)
    _hash_config: dict[int, tuple[type, Any, Any]] = {  # noqa: RUF012
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16bit configs do not use a mersenne prime number.
        # The original reason for using mersenne prime was speed.
        # Testing reveals there is no benefit to using a 2^61 mersenne prime number for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    @override
    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.bands is None and self.rows is None:
            # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
            # of probabilities of false positive and false negative, taken from datasketch.
            # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
            # The following assumes a "perfect hash". using 16bit hashes might challenge this assumption
            # lower precision dtype will cause more collisions, so higher false_positives and fewer false negatives.
            # Both effects move the result towards more documents being considered duplicates.
            self.bands, self.rows = optimal_param(
                self.threshold,
                self.num_perm,
                false_positive_weight=self.false_positive_weight,
                false_negative_weight=self.false_negative_weight,
            )

        DTYPE, MAX_HASH, MODULO_PRIME = self.get_hash_config()
        self._dtype = DTYPE
        self._modulo_prime = MODULO_PRIME
        self._max_hash = MAX_HASH

    def get_hash_config(self) -> tuple[type, Any, Any]:
        return self._hash_config[self.hash_bits]

    @property
    def hash_func(self) -> Callable[[bytes], int]:
        match self.hash_func_name:
            case "sha1":
                from text_dedup.utils.hashfunc import sha1_hash

                result = partial(sha1_hash, d=min(self.hash_bits, 32))
            case "xxh3":
                from text_dedup.utils.hashfunc import xxh3_hash

                result = partial(xxh3_hash, seed=self.seed, bits=min(self.hash_bits, 32))
        return result

    @property
    def hash_ranges(self) -> list[tuple[int, int]]:
        bands = self.bands or 0
        rows = self.rows or 0
        return [(i * rows, (i + 1) * rows) for i in range(bands)]

    @property
    def permutations(self) -> tuple[np.ndarray, np.ndarray]:
        """
        for minhash, we need to make a lot of hashes(=num_perms).
        In many previous implementations, this is achieved through a method described in
        `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
        There we start with a know good hash x (=hash_func) and permutate it as the following:
        `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
        the following produces these a, b pairs
        """
        if self._rng is None:
            raise ValueError("RNG is not materialized")  # noqa: TRY003

        # a is a multiplier so should not be 0
        a = self._rng.randint(1, self._modulo_prime, size=(self.num_perm,), dtype=self._dtype)
        b = self._rng.randint(0, self._modulo_prime, size=(self.num_perm,), dtype=self._dtype)

        return a, b

    def get_filtering_func(self) -> Callable[[str], bool]:
        tokenize_func = self._get_tokenize_func()

        def f(text: str) -> bool:
            return len(tokenize_func(text)) >= self.min_length

        return f

    def _get_tokenize_func(self) -> Callable[[str], list[str]]:
        NON_ALPHA = re.compile(r"\W", re.UNICODE)

        def f(content: str) -> list[str]:
            return NON_ALPHA.split(content.lower())

        return f

    def get_ngrams_func(self) -> Callable[[str], set[bytes]]:
        tokenize_func = self._get_tokenize_func()

        def f(content: str) -> set[bytes]:
            return {
                bytes(" ".join(t).lower(), "utf-8")
                for t in ngrams(tokenize_func(content), self.ngram_size, self.min_length)
            }

        return f

    def get_embed_func(
        self,
    ) -> Callable[[list[str], list[int]], dict[str, list[int | bytes]]]:
        """Create a function that embeds a string into a list of (index, hash, length) tuples."""
        # Fall back to pure Python implementation
        # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
        # the formula is a * x(base hash of each shingle) + b
        a, b = self.permutations
        hash_func = self.hash_func
        hash_ranges = self.hash_ranges
        ngrams_func = self.get_ngrams_func()

        def f(text_col: list[str], idx_col: list[int]) -> dict[str, list[int | bytes]]:
            # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
            # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
            content: str = text_col[0]
            idx: int = idx_col[0]
            tokens: set[bytes] = ngrams_func(content)
            hashvalues: npt.NDArray[Any] = np.array([hash_func(token) for token in tokens], dtype=self._dtype).reshape(  # pyright: ignore[reportUnknownVariableType]
                len(tokens), 1
            )
            # Permute the hash values to produce new universal hashes
            # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
            # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
            hashvalues = (hashvalues * a + b) % self._modulo_prime & self._max_hash
            # this part is where the name "min" of minhash comes from
            # this stacks all the hashes and then takes the minimum from each column
            masks: npt.NDArray[Any] = np.full(shape=self.num_perm, dtype=self._dtype, fill_value=self._max_hash)  # pyright: ignore[reportUnknownVariableType]
            hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
            # Originally, byteswap was done for speed. Testing shows it has a negligible impact
            # keeping for backward compatibility, even though theoretically and empirically
            # it doesn't matter if it is there or not. github.com/ekzhu/datasketch/issues/114
            return {
                "__band_idx__": list(range(len(hash_ranges))),
                "__band_val__": [bytes(hashvalues[start:end].byteswap().data) for (start, end) in hash_ranges],
                self.internal_index_column: [idx for _ in hash_ranges],
            }

        return f
