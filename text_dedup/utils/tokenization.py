#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:59:42
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from itertools import tee
from typing import List
from typing import Text


def ngrams(sequence: List[Text], n: int):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b"], 3))
    [['a', 'b']]
    """
    if len(sequence) < n:
        return iter([sequence])
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)
