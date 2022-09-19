#!/usr/bin/env python
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Sequence

import numpy as np
import xxhash

from text_dedup.base import Embedder
from text_dedup.base import Fingerprint
from text_dedup.preprocess import tokenize


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


def _compute(hashes: List[int]) -> int:
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
    >>> _compute([13352372148217134600, 5020219685658847592])
    18297957875485474664
    """
    # Convert integers to 64 bit binary arrays
    bits = np.unpackbits(np.array(hashes, dtype='u8').view(
        'u1').reshape(len(hashes), -1), axis=1).astype('i8')
    # Sum up each dimension of the arrays and take the sign
    counts = np.where(np.sum(2 * bits - 1, axis=0).astype(dtype='i8')
                      >= 0, 1, 0).astype('u1')
    # Convert the binary array back to an integer
    # Change to
    # return np.packbits(counts).view('>u8').item()s
    # to match the code in https://github.com/seomoz/simhash-cpp/, although it doesn't really matter
    return np.packbits(counts).view('u8').item()


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
        [15336018574513328062, 5455741392207466107]
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
        9996463820397055579
        """
        def wrapper(doc: str) -> Fingerprint:
            tokens = self.tokenizer(doc, **kwargs)
            ans = _compute([_unsigned_hash(t.encode('utf-8')) for t in tokens])
            return ans

        return wrapper

    def __repr__(self) -> str:
        return f'SimHashEmbedder(tokenizer={self.tokenizer.__name__})'
