#!/usr/bin/env python
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

from text_dedup.utils.tokenizer import tokenize


def _unsigned_hash(obj: bytes, bit_length: int = 64) -> int:
    """
    Compute a 64-bit hash of an object.

    This is a modified version of https://github.com/seomoz/simhash-py/blob/master/simhash/simhash.pyx.

    Parameters
    ----------
    obj : bytes
        The object to hash.
    bit length : int
        The bit length of the hash.

    Returns
    -------
    int
        The hash of the object.

    Examples
    --------
    >>> _unsigned_hash(b'hello world', 64)
    13352372148217134600
    """
    h = hashlib.sha256(obj).digest()[: bit_length // 8]
    return int.from_bytes(h, byteorder='big', signed=False)


def _compute(hashes: List[int], bit_length: int = 64) -> int:
    """
    Compute the Simhash of a list of hashes.

    Parameters
    ----------
    hashes : List[int]
        The list of hashes.
    bit_length : int
        The bit length of the hash.

    Returns
    -------
    int
        The Simhash of the list of hashes.

    Examples
    --------
    >>> _compute([13352372148217134600], 64)
    13352372148217134600
    """
    bits = np.unpackbits(np.array(hashes, dtype='>u8').view(
        '>u1').reshape(len(hashes), -1), axis=1).astype('>i8')
    counts = np.where(np.sum(2 * bits - 1, axis=0, dtype='>i8') >= 0, 1, 0).astype('>u1')
    return np.packbits(counts).view('>u8').item()


@dataclass
class SimHashEmbedder:

    """
    Embedding text using SimHash.
    """

    def embed(self, corpus: List[str], **kwargs) -> List[int]:
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
        List[int]
            Fingerprints of the corpus.

        Examples
        --------
        >>> embedder = SimHashEmbedder()
        >>> embeddings = embedder.embed(["hello", "hello world! This is a test."])
        >>> embeddings
        [4051436901025898700, 13943988908483280899]
        """
        f = self.embed_function(**kwargs)
        return [f(doc) for doc in corpus]

    @staticmethod
    def embed_function(**kwargs) -> Callable:
        """
        Embedding function that takes a string and returns the embedding/fingerprint.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for tokenization.

        Examples
        --------
        >>> embedder = SimHashEmbedder()
        >>> hashes = embedder.embed_function()("hello world! This is a test string.")
        >>> hashes
        13950746197979717635
        """
        use_str = kwargs.pop('use_str', False)

        def wrapper(doc: str) -> Union[int, str]:
            tokens = tokenize(doc, **kwargs)
            ans = _compute(
                list(
                    map(
                        lambda x: _unsigned_hash(
                            x.encode('utf-8'),
                        ),
                        tokens,
                    ),
                ),
            )
            if use_str:
                return str(ans)
            return ans

        return wrapper

    def __repr__(self) -> str:

        return 'SimHashEmbedder()'
