#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-05-22 11:33:39
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from typing import List
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def tokenize(text: str, n_gram: int = 6, level: str = 'word', lan: str = "eng") -> List[str]:
    """
    Tokenize the text into a sequence of strings.

    Parameters
    ----------
    text : str
        The text to tokenize.
    n_gram : int, optional (default=6)
        The size of the n-grams to use.
    level : str, optional (default='word')
        The level of tokenization to use.
    lan : str, optional (default='eng')
        The language of the text.

    Returns
    -------
    List[str]
        The list of tokens.

    Examples
    --------
    >>> tokenize("This is a test.", n_gram=2)
    ['This is', 'is a', 'a test', 'test .']
    >>> tokenize("This is a test.", n_gram=2, level='char')
    ['Th', 'hi', 'is', 's ', ' i', 'is', 's ', ' a', 'a ', ' t', 'te', 'es', 'st', 't.']
    """

    assert level in {'word', 'char'}, f"Invalid level: {level}"
    assert lan in {'eng',}, f"Invalid language: {lan}"

    if lan == "eng":
        if level == "word":
            return [" ".join(ngram) for ngram in ngrams(word_tokenize(text), n=n_gram)]
        elif level == "char":
            return ["".join(ngram) for ngram in ngrams(text, n=n_gram)]
    
    return []
    

