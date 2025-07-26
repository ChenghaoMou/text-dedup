import os
from collections import defaultdict
from functools import partial
from typing import Any
from typing import Callable
from typing import Literal
from typing import TypeAlias

import numpy as np
import regex as re
from pydantic_settings import BaseSettings

from text_dedup.utils.analysis import optimal_param
from text_dedup.utils.tokenization import ngrams


class AlgorithmConfig(BaseSettings):
    algorithm_name: Literal["minhash", "simhash", "bloomfilter"]
    text_column: str
    index_column: str | None = None
    cluster_column: str = "__CLUSTER__"
    signature_column: str = "__SIGNATURE__"
    seed: int = 42
    num_proc: int = max(1, os.cpu_count() or 1)
    batch_size: int = 1000
    _rng: np.random.RandomState | None
    _internal_index_column: str = "__INDEX__"

    def model_post_init(self, context: Any) -> None:
        self._rng = np.random.RandomState(self.seed)

    @property
    def internal_index_column(self) -> str:
        return self._internal_index_column


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
    _hash_config: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16bit configs do not use a mersenne prime number.
        # The original reason for using mersenne prime was speed.
        # Testing reveals there is no benefit to using a 2^61 mersenne prime number for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

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

    def create_hash_tables(self) -> list[dict[int, set]]:
        return [defaultdict(set) for _ in range(self.bands or 0)]

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

    def get_filtering_func(self) -> Callable[[dict[str, Any]], bool]:
        tokenize_func = self._get_tokenize_func()

        def f(record: dict[str, Any]) -> bool:
            return len(tokenize_func(record[self.text_column])) >= self.min_length

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
    ) -> Callable[[str, int], dict[str, Any]]:
        """
        Calculate hash values for the content.

        Parameters
        ----------
        content : str
            The content to be embedded.
        idx : int
            The index of the content.
        use_rust : bool, optional
            Whether to use Rust acceleration if available, True by default.

        Returns
        -------
        Dict[str, Any]
            The hash values in each range and the index.

        Examples
        --------
        >>> content = "hello world"
        >>> idx = 0
        >>> num_perm = 250
        >>> ngram_size = 1
        >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
        >>> max_hash = np.uint32((1 << 32) - 1)
        >>> modulo_prime = np.uint32((1 << 32) - 5)
        >>> PERMUTATIONS = (RNG.randint(1, modulo_prime, size=num_perm), RNG.randint(0, modulo_prime, size=num_perm))
        >>> res = embed_func(
        ...     content,
        ...     idx,
        ...     num_perm=num_perm,
        ...     ngram_size=ngram_size,
        ...     min_length=0,
        ...     hashranges=hashranges,
        ...     permutations=PERMUTATIONS,
        ...     hash_func=xxh3_32hash,
        ...     dtype=np.uint32,
        ...     max_hash=max_hash,
        ...     modulo_prime=modulo_prime,
        ... )
        >>> len(res[SIGNATURE_COLUMN])
        10
        >>> res[INDEX_COLUMN]
        0
        """
        # Fall back to pure Python implementation
        # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
        # the formula is a * x(base hash of each shingle) + b
        a, b = self.permutations
        hash_func = self.hash_func
        hash_ranges = self.hash_ranges
        ngrams_func = self.get_ngrams_func()

        def f(content: str, idx: int) -> dict[str, Any]:
            # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
            # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
            tokens: set[bytes] = ngrams_func(content)
            hashvalues: np.ndarray = np.array([hash_func(token) for token in tokens], dtype=self._dtype).reshape(
                len(tokens), 1
            )
            # Permute the hash values to produce new universal hashes
            # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
            # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
            hashvalues = (hashvalues * a + b) % self._modulo_prime & self._max_hash
            # this part is where the name "min" of minhash comes from
            # this stacks all the hashes and then takes the minimum from each column
            masks: np.ndarray = np.full(shape=self.num_perm, dtype=self._dtype, fill_value=self._max_hash)
            hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
            # Originally, byteswap was done for speed. Testing shows it has a negligible impact
            # keeping for backward compatibility, even though theoretically and empirically
            # it doesn't matter if it is there or not. github.com/ekzhu/datasketch/issues/114
            Hs: list[bytes] = [bytes(hashvalues[start:end].byteswap().data) for start, end in hash_ranges]
            return {self.signature_column: Hs, self.internal_index_column: idx}

        return f


class SimHashAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["simhash"] = "simhash"


class BloomFilterAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["bloomfilter"] = "bloomfilter"


AlgoConfig: TypeAlias = MinHashAlgorithmConfig | SimHashAlgorithmConfig | BloomFilterAlgorithmConfig
