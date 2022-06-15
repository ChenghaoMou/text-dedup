#!/usr/bin/env python
# @Date    : 2022-04-02 10:53:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
# from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from datasketch import MinHash

from text_dedup.utils.tokenizer import tokenize


@dataclass
class MinHashEmbedder:
    """
    Embedding text using MinHash.

    Parameters
    ----------
    num_perm : int, optional (default=128)
        Number of permutations to use.
    """

    num_perm: int = 128

    def embed(self, corpus: List[str], **kwargs) -> np.ndarray:
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
        np.ndarray
            Embedding of the corpus.

        Examples
        --------
        >>> embedder = MinHashEmbedder(128)
        >>> embeddings = embedder.embed(["hello world", "hello world"])
        >>> embeddings.shape
        (2, 128)
        """
        f = self.embed_function(**kwargs)
        return np.array([f(doc) for doc in corpus])

    def embed_function(self, **kwargs) -> Callable:
        """
        Embedding function that takes a string and returns the embedding/fingerprint.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for tokenization.

        Returns
        -------
        Callable
            Embedding function.

        Examples
        --------
        >>> embedder = MinHashEmbedder(128)
        >>> hashes = embedder.embed_function()("hello world")
        >>> hashes.shape
        (128,)
        """

        def wrapper(doc: str) -> np.ndarray:
            m = MinHash(num_perm=self.num_perm)
            for ngram in tokenize(doc, **kwargs):
                m.update(ngram.encode('utf-8'))
            return m.hashvalues

        return wrapper
