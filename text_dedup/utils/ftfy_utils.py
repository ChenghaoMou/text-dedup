#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:37:44
# @Author  : Chris Ha (hac541309@gmail.com) Chenghao Mou (mouchenghao@gmail.com)

from typing import Dict
from typing import List

from ftfy import TextFixerConfig
from ftfy import apply_plan
from ftfy import fix_and_explain
from ftfy import fix_encoding
from ftfy import fix_text


def fix_text_batch(batch: Dict[str, List], column: str = "text") -> Dict[str, List]:
    """
    Passes a batch of strings from batch[column] through fix_text
    uses the settings of default TextFixerConfig()
    https://ftfy.readthedocs.io/en/latest/config.html
    This is not concurrently processed, but batching will reduce mp overhead.

    Individualized configurations might be required depending on corpora.
    use fix_and_explain() for this

    Parameters
    ----------
    batch : Dict[str, List]
        The batch to be fixed with fix_text.

    Returns
    -------
    batch : Dict[str, List]
        The batch fixed with fix_text.

    Examples
    --------
    >>> batch = {"text":["hello world",r"L&AMP;AMP;ATILDE;&AMP;AMP;SUP3;PEZ",]}
    >>> fix_text_batch(batch,column="text")
    {'text': ['hello world', 'LóPEZ']}
    """
    batch[column] = [fix_text(text) for text in batch[column]]

    return batch


def fix_encoding_batch(batch: Dict[str, List], column: str = "text") -> Dict[str, List]:
    """
    Passes a batch of strings from batch[column] through fix_encoding.
    This includes fixing text by encoding and decoding it in different encodings,
    as well as the subordinate fixes `restore_byte_a0`, `replace_lossy_sequences`,
    `decode_inconsistent_utf8`, and `fix_c1_controls`.
    This is not concurrently processed, but batching will reduce mp overhead.

    When handling code, JSON,HTML or otherwise non natural language text, this might be better.
    As with fix_text, test on your target corpora before making a decision.

    Parameters
    ----------
    batch : Dict[str, List]
        The batch to be fixed with fix_encode.

    Returns
    -------
    batch : Dict[str, List]
        The batch fixed with fix_encode.

    Examples
    --------
    >>> batch = {"text":["hello world","voilÃ le travail",]}
    >>> fix_encoding_batch(batch,column="text")
    {'text': ['hello world', 'voilà le travail']}
    """
    batch[column] = [fix_encoding(text) for text in batch[column]]

    return batch


DEFAULT_TextFixerConfig = TextFixerConfig()
"""
DEFAULT_TextFixerConfig = ftfy.TextFixerConfig()
>>> DEFAULT_TextFixerConfig
>>> TextFixerConfig(
    unescape_html="auto",
    remove_terminal_escapes=True,
    fix_encoding=True,
    restore_byte_a0=True,
    replace_lossy_sequences=True,
    decode_inconsistent_utf8=True,
    fix_c1_controls=True,
    fix_latin_ligatures=True,
    fix_character_width=True,
    uncurl_quotes=True,
    fix_line_breaks=True,
    fix_surrogates=True,
    remove_control_chars=True,
    normalization="NFC",
    max_decode_length=1000000,
    explain=True,
)
"""


__all__ = [
    "fix_text",
]
