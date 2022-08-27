#!/usr/bin/env python
# @Date    : 2022-04-02 10:53:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from datasketch import MinHash

from text_dedup.preprocess.tokenizer import tokenize


@dataclass
class MinHashEmbedder:
    """
    Old but gold MinHash. This is basically a wrapper around the datasketch library.

    Parameters
    ----------
    num_perm : int, optional (default=128)
        Number of permutations to use.
    seed : int, optional (default=42)
        Seed for the random number generator and permutation.
    tokenizer : Callable, optional (default=tokenize)
        Tokenizer function.

    Examples
    --------
    >>> from text_dedup.embedders.minhash import MinHashEmbedder
    >>> embedder = MinHashEmbedder(128)
    """

    num_perm: int = 128
    seed: int = 42
    tokenizer: Callable[..., Tuple[List[str], List[Tuple[int, int]]]] = tokenize

    def embed(self, corpus: List[str], **kwargs) -> List[np.ndarray]:
        """
        Embed a list of strings. It applies the embedding function to each string sequentially.
        It is recommended to use the `embed_function` method instead, in parallel, for example.

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
        >>> from text_dedup.embedders.minhash import MinHashEmbedder
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
        >>> from text_dedup.embedders.minhash import MinHashEmbedder
        >>> embedder = MinHashEmbedder(128)
        >>> hashes = embedder.embed_function()("hello world")
        >>> hashes.shape
        (128,)
        """
        # This is only needed for simhash but here we do it for consistency
        kwargs.pop("use_str", None)

        def wrapper(doc: str) -> np.ndarray:
            m: MinHash = MinHash(num_perm=self.num_perm, seed=self.seed)
            tokens, _ = self.tokenizer(doc, **kwargs)
            for ngram in tokens:
                m.update(ngram.encode("utf-8"))
            return m.hashvalues

        return wrapper

    def __repr__(self) -> str:

        return f"MinHashEmbedder(num_perm={self.num_perm}, seed={self.seed}, tokenizer={self.tokenizer.__name__})"
