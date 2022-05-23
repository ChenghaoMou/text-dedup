#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 10:53:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)


from dataclasses import dataclass
from typing import List

import numpy as np
from datasketch import MinHash
from text_dedup.embedders import Embedder
from text_dedup.utils.tokenizer import tokenize


@dataclass
class MinHashEmbedder(Embedder):
    """
    Embedding text using MinHash.

    Parameters
    ----------
    num_perm : int, optional (default=128)
        Number of permutations to use.
    ngram_size : int, optional (default=5)
        Size of ngrams to use.
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
        """
        signatures: List[np.ndarray] = []
        for doc in corpus:
            m = MinHash(num_perm=self.num_perm)
            for ngram in tokenize(doc, **kwargs):
                m.update(ngram.encode("utf-8"))
            signatures.append(m.hashvalues)

        return np.asarray(signatures)
    
    def embed_function(self, **kwargs):
        def wrapper(doc: str) -> np.ndarray:
            m = MinHash(num_perm=self.num_perm)
            for ngram in tokenize(doc, **kwargs):
                m.update(ngram.encode("utf-8"))
            return m.hashvalues
        return wrapper
