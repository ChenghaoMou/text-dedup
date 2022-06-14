#!/usr/bin/env python
# @Date    : 2022-05-22 11:33:39
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

from itertools import tee
from typing import List

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')


def ngrams(sequence: List[str], n: int):
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def tokenize(text: str, n_gram: int = 6, level: str = 'sentencepiece') -> List[str]:
    """
    Tokenize the text into a sequence of strings.

    Parameters
    ----------
    text : str
        The text to tokenize.
    n_gram : int, optional (default=6)
        The size of the n-grams to use.
    level : str, optional (default='sentencepiece')
        The level of tokenization to use.

    Returns
    -------
    List[str]
        The list of tokens.

    Examples
    --------
    >>> tokenize("This is a test.", n_gram=2)
    ['▁This▁is', '▁is▁', '▁a', 'a▁test', '▁test.']
    >>> tokenize("This is a test.", n_gram=2, level='char')
    ['Th', 'hi', 'is', 's ', ' i', 'is', 's ', ' a', 'a ', ' t', 'te', 'es', 'st', 't.']
    """

    assert level in {'sentencepiece', 'char'}, f'Invalid level: {level}'

    if level == 'sentencepiece':
        return [''.join(ngram) for ngram in ngrams(tokenizer.tokenize(text), n=n_gram)]
    elif level == 'char':
        return [''.join(ngram) for ngram in ngrams(list(text), n=n_gram)]

    return []
