#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import hashlib
import struct
from dataclasses import dataclass
from typing import List

import numpy as np
from text_dedup.embedders import Embedder
from text_dedup.utils.tokenizer import tokenize


def _unsigned_hash(obj: bytes):
    """Returns a hash suitable for use as a hash_t. source: https://github.com/seomoz/simhash-py/blob/master/simhash/simhash.pyx"""
    # Takes first 8 bytes of MD5 digest
    digest = hashlib.md5(obj).digest()[0:8]
    # Unpacks the binary bytes in digest into a Python integer
    return struct.unpack(">Q", digest)[0] & 0xFFFFFFFFFFFFFFFF


def _compute(hashes: List[int]) -> int:
    """Computes the simhash of a list of hashes."""
    counts = np.zeros(64, dtype=np.int64)
    for h in hashes:
        for i in range(64):
            counts[i] += (h & 1) * 2 - 1
            h >>= 1

    result = 0
    for i in range(64):
        if counts[i] > 0:
            result |= 1 << i
    return result


@dataclass
class SimHashEmbedder(Embedder):

    """
    Embedding text using SimHash.

    Parameters
    ----------
    num_perm : int, optional (default=128)
        Number of permutations to use.
    threshold : float, optional (default=0.5)
        Threshold for similarity.
    """

    num_perm: int = 128
    threshold: float = 0.5

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
        """
        signatures: List[int] = []
        for doc in corpus:
            tokens = tokenize(doc, **kwargs)
            signatures.append(
                _compute(
                    map(lambda x: _unsigned_hash(" ".join(x).encode("utf-8")), tokens)
                )
            )

        return signatures

    def embed_function(self, **kwargs) -> int:
        """
        Embedding function that takes a string and returns the embedding/fingerprint.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for tokenization.

        Returns
        -------
        int
            Fingerprint of the string.
        """

        def wrapper(doc: str) -> int:

            tokens = tokenize(doc, **kwargs)
            return _compute(
                map(lambda x: _unsigned_hash(" ".join(x).encode("utf-8")), tokens)
            )

        return wrapper
