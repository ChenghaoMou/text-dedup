#!/usr/bin/env python
# @Date    : 2022-05-22 11:33:39
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import re
from typing import Any
from typing import List
from typing import Literal

from transformers import XLMRobertaTokenizerFast

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
# this is from https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/minhash_deduplication.py
NON_ALPHA = re.compile("[^A-Za-z_0-9]")


def ngrams(sequence: List[Any], n: int, step_size: int = -1) -> List[List[Any]]:
    """Generate n-grams from a sequence.

    Parameters
    ----------
    sequence : List[Any]
        List of elements.
    n : int
        The size of the n-grams to use.
    step_size : int, optional (default=-1)
        The step size to use when generating n-grams. If -1, then the step size is the same as the n-gram size.

    Returns
    -------
    List[List[Any]]
        The list of n-grams.

    Examples
    --------
    >>> from text_dedup.preprocess.tokenization import ngrams
    >>> list(ngrams(['a', 'b', 'c'], n=1))
    [['a'], ['b'], ['c']]
    >>> list(ngrams(['a', 'b', 'c'], n=6))
    [['a', 'b', 'c']]
    >>> list(ngrams(['a', 'b', 'c'], n=2))
    [['a', 'b'], ['c']]
    >>> list(ngrams(['a', 'b', 'c'], n=2, step_size=1))
    [['a', 'b'], ['b', 'c'], ['c']]
    """
    if len(sequence) <= n:
        return [sequence]

    results = []
    for i in range(0, len(sequence), step_size if step_size != -1 else n):
        results.append(sequence[i: i + n])
    return results


def tokenize(
        text: str,
        n_gram: int = 6,
        level: Literal["sentencepiece", "char", "word"] = 'sentencepiece',
        step_size: int = -1,
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
    step_size : int, optional (default=-1)
        The step size to use when generating n-grams. If -1, then the step size is the same as the n-gram size.

    Returns
    -------
    List[str]
        The list of tokens, and the list of token boundaries.

    Examples
    --------
    >>> tokenize("Hello world!")
    ['▁Hello▁world!']
    >>> tokenize("This is a test.", n_gram=2)
    ['▁This▁is', '▁a▁test', '.']
    >>> tokenize("test message", n_gram=2, level='char')
    ['te', 'st', ' m', 'es', 'sa', 'ge']
    """
    if level == 'sentencepiece':
        tokens = tokenizer.tokenize(text)
    elif level == 'char':
        tokens = list(text)
    elif level == "word":
        tokens = [w for w in text.split(" ") if w]
    elif level == "code":
        tokens = [t for t in NON_ALPHA.split(text) if t.strip()]
    else:
        raise ValueError(f"Invalid level: {level}")

    if n_gram == 1 and step_size == -1:
        return tokens

    output_tokens = []

    for window_tokens in ngrams(tokens, n_gram, step_size):
        output_tokens.append(''.join(window_tokens))

    return output_tokens
