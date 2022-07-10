#!/usr/bin/env python
# @Date    : 2022-04-02 10:53:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from dataclasses import dataclass
from typing import Callable, List

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

    def embed(self, corpus: List[str], **kwargs) -> List[np.ndarray]:
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
        List[np.ndarray]
            Embeddings of the corpus.

        Examples
        --------
        >>> embedder = MinHashEmbedder(128)
        >>> embeddings = embedder.embed(["hello world", "hello world"])
        >>> len(embeddings)
        2
        """
        f = self.embed_function(**kwargs)
        return [f(doc) for doc in corpus]

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
        _ = kwargs.pop("use_str", False)

        def wrapper(doc: str) -> np.ndarray:
            m = MinHash(num_perm=self.num_perm)
            tokens, _ = tokenize(doc, **kwargs)
            for ngram in tokens:
                m.update(ngram.encode("utf-8"))
            return m.hashvalues

        return wrapper

    def __repr__(self) -> str:

        return f"MinHashEmbedder(num_perm={self.num_perm})"
