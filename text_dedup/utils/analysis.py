#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-01-02 15:18:55
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List

from scipy.integrate import quad as integrate

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


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.

    Examples
    --------
    >>> optimal_param(0.75, 256)
    (21, 12)
    >>> optimal_param(0.75, 256, 0.1, 0.9)
    (28, 9)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt
