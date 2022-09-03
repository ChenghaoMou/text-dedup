#!/usr/bin/env python
# @Date    : 2022-05-22 11:33:39
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Any, List, NamedTuple, Tuple

from transformers import XLMRobertaTokenizerFast

# Offset = NamedTuple("Offset", [("start", int), ("end", int)])
Offset = Tuple[int, int]
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


def tokenize(text: str, n_gram: int = 6, level: str = 'sentencepiece') -> Tuple[List[str], List[Tuple[int, int]]]:
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
    Tuple[List[str], List[Tuple[int, int]]]
        The list of tokens, and the list of token boundaries.

    Examples
    --------
    >>> tokenize("Hello world!")
    (['▁Hello▁world!'], [(0, 12)])
    >>> tokenize("This is a test.", n_gram=2)
    (['▁This▁is', '▁is▁a', '▁a▁test', '▁test.'], [(0, 7), (5, 9), (8, 14), (10, 15)])
    >>> tokenize("test message", n_gram=2, level='char')
    (['te', 'es', 'st', 't ', ' m', 'me', 'es', 'ss', 'sa', 'ag', 'ge'], [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12)])
    """

    assert level in {'sentencepiece', 'char', 'word'}, f'Invalid level: {level}'

    if level == 'sentencepiece':
        tokens = tokenizer.tokenize(text)
        offsets = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=False).pop("offset_mapping")
    elif level == 'char':
        tokens = list(text)
        offsets = []
        for token in tokens:
            if not offsets:
                offsets.append((0, len(token)))
                continue
            offsets.append((offsets[-1][1], offsets[-1][1] + len(token)))
    elif level == "word":
        tokens = text.split(" ")
        offsets = []
        for token in tokens:
            if not offsets:
                offsets.append((0, len(token)))
                continue
            offsets.append((offsets[-1][1] + 1, offsets[-1][1] + len(token) + 1))

    output_tokens = []
    output_offsets = []

    for window_tokens, window_offsets in zip(ngrams(tokens, n_gram), ngrams(offsets, n_gram)):
        output_tokens.append(''.join(window_tokens))
        output_offsets.append((window_offsets[0][0], window_offsets[-1][1]))

    return output_tokens, output_offsets
