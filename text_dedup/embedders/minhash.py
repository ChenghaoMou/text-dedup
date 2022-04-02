#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 10:53:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)


from dataclasses import dataclass
from typing import List

import numpy as np
from datasketch import MinHash
from nltk.util import ngrams

from text_dedup.embedders import Embedder


@dataclass
class MinHashEmbedder(Embedder):

    num_perm: int = 128
    ngram_size: int = 5

    def embed(self, corpus: List[str]) -> np.ndarray:

        signatures: List[np.ndarray] = []
        for doc in corpus:
            m = MinHash(num_perm=self.num_perm)
            for ngram in ngrams(doc, self.ngram_size):
                m.update(" ".join(ngram).encode("utf-8"))
            signatures.append(m.hashvalues)

        return np.asarray(signatures)
