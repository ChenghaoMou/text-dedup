#!/usr/bin/env python
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Sequence

import numpy as np
import xxhash

from text_dedup.base import Embedder
from text_dedup.base import Fingerprint
from text_dedup.preprocess import tokenize

BIT_MASK: np.ndarray = 2 ** np.arange(64, dtype=np.uint64).reshape([1, 64])


def unpackbits(x: np.ndarray, num_bits: int = 64) -> np.ndarray:
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    return (x & BIT_MASK).astype(bool).astype(int).reshape(xshape + [num_bits])


def _unsigned_hash(obj: bytes) -> int:
    """
    Compute a 64-bit hash of an object.

    It doesn't really matter what hash function to use, as long as it is consistent.

    Parameters
    ----------
    obj: bytes
        The object to hash.

    Returns
    -------
    int
        The hash of the object.

    Examples
    --------
    >>> _unsigned_hash(b'hello world')
    5020219685658847592
    """
    return xxhash.xxh64(obj).intdigest()

    # digest = hashlib.md5(obj).digest()[:8]
    # # Unpacks the binary bytes in digest into a Python integer
    # return struct.unpack('>Q', digest)[0] & 0xFFFFFFFFFFFFFFFF


def compute(hashes: List[int]) -> int:
    """
    Compute the Simhash of a list of hashes.

    Notes to myself: You tried porting this to Cython, but it didn't improve the performance.

    Parameters
    ----------
    hashes : List[int]
        The list of hashes.

    Returns
    -------
    int
        The Simhash of the list of hashes.

    Examples
    --------
    >>> compute([13352372148217134600, 5020219685658847592])
    74633958390507528
    """
    bits = 2 * unpackbits(np.asarray(hashes, dtype=np.uint64), 64) - 1
    res = (np.where(np.sum(bits, axis=0) > 0, 1, 0)[::-1]).astype(np.uint64)
    return np.packbits(res).view(np.uint64).byteswap().item()


@dataclass
class SimHashEmbedder(Embedder):
    """
    Embedding text based on `SimHash <https://bit.ly/3TLgzQv>`.

    Parameters
    ----------
    tokenizer : Callable, optional (default=tokenize)
        Tokenizer function.

    Examples
    --------
    >>> from text_dedup.near_dedup.simhash import SimHashEmbedder
    >>> embedder = SimHashEmbedder()
    """

    tokenizer: Callable[..., List[str]] = tokenize

    def embed(self, corpus: Sequence[str], **kwargs) -> List[Fingerprint]:
        """
        Embed a list of strings.

        Parameters
        ----------
        corpus : List[str]
            List of strings to embed.
        kwargs : dict
            Additional keyword arguments for tokenization.

        Returns
        -------
        List[Singature]
            Fingerprints of the corpus.

        Examples
        --------
        >>> embedder = SimHashEmbedder()
        >>> embeddings = embedder.embed(["hello", "hello world! This is a test."])
        >>> embeddings
        [15336018574513328062, 3300884242954]
        """
        f = self.embed_function(**kwargs)
        return [f(doc) for doc in corpus]

    def embed_function(self, **kwargs) -> Callable[[str], Fingerprint]:
        """
        Embedding function that takes a string and returns the embedding/fingerprint.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for tokenization.

        Returns
        -------
        Callable[[str], Fingerprint]
            Embedding function.

        Examples
        --------
        >>> embedder = SimHashEmbedder()
        >>> hashes = embedder.embed_function()("hello world! This is a test string.")
        >>> hashes
        49566403822960672
        """
        def wrapper(doc: str) -> Fingerprint:
            tokens = self.tokenizer(doc, **kwargs)
            ans = compute([_unsigned_hash(t.encode('utf-8')) for t in tokens])
            return ans

        return wrapper

    def __repr__(self) -> str:
        return f'SimHashEmbedder(tokenizer={self.tokenizer.__name__})'
