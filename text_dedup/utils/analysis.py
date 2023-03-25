#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-01-02 15:18:55
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List

from text_dedup.utils.tokenization import ngrams


def jaccard_similarity(
    doc1: str | List[str],
    doc2: str | List[str],
    ngram_size: int = 8,
    min_length: int = 0,
) -> float:
    """Compute the Jaccard similarity between two documents.

    Parameters
    ----------
    doc1 : str or List[str]
        The first document.
    doc2 : str or List[str]
        The second document.
    ngram_size : int, optional
        The size of n-grams, by default 8
    min_length : int, optional
        The minimum length of each n-gram, by default 0

    Returns
    -------
    float
        The Jaccard similarity.

    Examples
    --------
    >>> jaccard_similarity("hello world", "hello world")
    1.0
    >>> jaccard_similarity("hello world", "hello world!")
    0.8
    >>> jaccard_similarity("hello world".split(), "hello world!".split(), ngram_size=1)
    0.3333333333333333
    """
    words1 = set(" ".join(ng) for ng in ngrams(list(doc1), ngram_size, min_length=min_length))
    words2 = set(" ".join(ng) for ng in ngrams(list(doc2), ngram_size, min_length=min_length))
    return len(words1 & words2) / max(1, len(words1 | words2))
