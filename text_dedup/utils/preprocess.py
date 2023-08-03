#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-05-06 19:39:27
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import regex as re

DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(r"[\p{P}\p{C}\p{S}]+")


def normalize(line: str) -> str:
    """
    Normalize a line of text. Source: https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/text_normalizer.py#L180

    Parameters
    ----------
    line : str
        The line of text to normalize.

    Returns
    -------
    str
        The normalized line of text.

    Examples
    --------
    >>> normalize("Hello, world!")
    'hello world'
    >>> normalize("Hello, 123!\\n\\t\\b")
    'hello 000'
    """
    line = line.strip()
    if not line:
        return line
    line = line.lower()
    line = DIGIT_RE.sub("0", line)
    line = PUNCT_OR_NON_PRINTING_CHARS_RE.sub("", line)
    return line
