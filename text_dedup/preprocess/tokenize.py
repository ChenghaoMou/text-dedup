#!/usr/bin/env python
# @Date    : 2022-05-22 11:33:39
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Any
from typing import List
from typing import Literal

from transformers import XLMRobertaTokenizerFast

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True


def ngrams(sequence: List[Any], n: int) -> List[List[Any]]:
    """Generate n-grams from a sequence.

    Parameters
    ----------
    sequence : List[Any]
        List of elements.
    n : int
        The size of the n-grams to use.

    Returns
    -------
    List[List[Any]]
        The list of n-grams.

    Examples
    --------
    >>> list(ngrams(['a', 'b', 'c'], n=1))
    [['a'], ['b'], ['c']]
    >>> list(ngrams(['a', 'b', 'c'], n=6))
    [['a', 'b', 'c']]
    """
    if len(sequence) <= n:
        return [sequence]

    results = []
    for i in range(len(sequence) - n + 1):
        results.append(sequence[i: i + n])
    return results


def tokenize(
        text: str,
        n_gram: int = 6,
        level: Literal["sentencepiece", "char", "word"] = 'sentencepiece'
) -> List[str]:
    """
    Tokenize the text into a sequence of strings.

    Parameters
    ----------
    text : str
        The text to tokenize.
    n_gram : int, optional (default=6)
        The size of the n-grams to use.
    level : Literal["sentencepiece", "char", "word"], optional (default='sentencepiece')
        The level of tokenization to use.

    Returns
    -------
    List[str]
        The list of tokens, and the list of token boundaries.

    Examples
    --------
    >>> tokenize("Hello world!")
    ['▁Hello▁world!']
    >>> tokenize("This is a test.", n_gram=2)
    ['▁This▁is', '▁is▁a', '▁a▁test', '▁test.']
    >>> tokenize("test message", n_gram=2, level='char')
    ['te', 'es', 'st', 't ', ' m', 'me', 'es', 'ss', 'sa', 'ag', 'ge']
    """
    if level == 'sentencepiece':
        tokens = tokenizer.tokenize(text)
    elif level == 'char':
        tokens = list(text)
    elif level == "word":
        tokens = text.split(" ")
    else:
        raise ValueError(f"Invalid level: {level}")

    output_tokens = []

    for window_tokens in ngrams(tokens, n_gram):
        output_tokens.append(''.join(window_tokens))

    return output_tokens
