#!/usr/bin/env python
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from text_dedup.embedders import Embedder
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
    """
    assert bit_length == 64, 'Only 64-bit hashes are supported.'
    h = hashlib.sha256(obj).digest()[: bit_length // 8]
    return int.from_bytes(h, byteorder='big', signed=False)


def _compute(hashes: list[int], bit_length: int = 64) -> int:
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
    """
    assert bit_length == 64, 'Only 64-bit hashes are supported.'
    counts = np.zeros(bit_length, dtype=np.int64)
    for h in hashes:
        i = 0
        while i < bit_length and h:
            counts[i] += (h & 1) * 2 - 1
            h >>= 1
            i += 1

    result = 0
    for i in range(bit_length):
        if counts[i] > 0:
            result |= 1 << i
    return result


@dataclass
class SimHashEmbedder(Embedder):

    """
    Embedding text using SimHash.
    """

    def embed(self, corpus: list[str], **kwargs) -> list[int]:
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
        """

        def wrapper(doc: str) -> int:

            tokens = tokenize(doc, **kwargs)
            return _compute(
                list(
                    map(
                        lambda x: _unsigned_hash(
                            ' '.join(x).encode('utf-8'),
                        ), tokens,
                    ),
                ),
            )

        return wrapper
