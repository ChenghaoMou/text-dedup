#!/usr/bin/env python
# @Date    : 2022-04-02 11:08:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from text_dedup.embedders.base import Embedder, Fingerprint
from text_dedup.preprocess.tokenizer import Offset, tokenize


def _unsigned_hash(obj: bytes, bit_length: int = 64) -> int:
    """
    Compute a 64-bit hash of an object.

    This is a modified version of https://github.com/seomoz/simhash-py/blob/master/simhash/simhash.pyx.

    Parameters
    ----------
    obj : bytes
        The object to hash.
    bit_length : int
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


def _compute(hashes: List[int]) -> int:
    """
    Compute the Simhash of a list of hashes.

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
    >>> _compute([13352372148217134600])
    13352372148217134600
    """
    bits = np.unpackbits(np.array(hashes, dtype='>u8').view(
        '>u1').reshape(len(hashes), -1), axis=1).astype('>i8')
    counts = np.where(np.sum(2 * bits - 1, axis=0, dtype='>i8') >= 0, 1, 0).astype('>u1')
    return np.packbits(counts).view('>u8').item()


@dataclass
class SimHashEmbedder(Embedder):
    """
    Embedding text based on `SimHash <https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf>`_.

    Parameters
    ----------
    tokenizer : Callable, optional (default=tokenize)
        Tokenizer function.

    Examples
    --------
    >>> from text_dedup.embedders import SimHashEmbedder
    >>> embedder = SimHashEmbedder()
    """

    tokenizer: Callable[..., Tuple[List[str], List[Offset]]] = tokenize

    def embed(self, corpus: List[str], **kwargs) -> List[Fingerprint]:
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
        [15473702421686509265, 16678727103752857983]
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
        14143049876155195771
        """
        # This is needed because datasets' (arrow) multiprocessing does not pickle int64 values
        # We need to convert it to str first
        use_str = kwargs.pop('use_str', False)

        def wrapper(doc: str) -> Fingerprint:
            tokens, _ = self.tokenizer(doc, **kwargs)
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
        return f'SimHashEmbedder(tokenizer={self.tokenizer.__name__})'
