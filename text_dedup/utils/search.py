#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-07 19:39:35
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import logging
import os
from collections import Counter
from typing import List, Tuple

from tqdm import tqdm

from text_dedup.preprocess.tokenizer import tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger: logging.Logger = logging.getLogger("text_dedup")


class BestFuzzyMatcher:
    def __init__(self, query_tokens: List[str], init_doc_tokens: List[str]) -> None:
        """
        Find the best matching window for the query in the document based on Jaccard similarity.

        Parameters
        ----------
        query_tokens : List[str]
            The query tokens.
        init_doc_tokens : List[str]
            The initial window tokens.

        Examples
        --------
        >>> query_tokens = ['a', 'b', 'c']
        >>> init_doc_tokens = ['e', 'a', 'b']
        >>> matcher = BestFuzzyMatcher(query_tokens, init_doc_tokens)
        >>> matcher.jaccard_similarity
        0.5
        >>> matcher.update('c', 'e')
        1.0
        >>> matcher.update('d', 'a')
        0.5
        """
        self.A = Counter(query_tokens)
        self.B = Counter(init_doc_tokens)
        self.intersection = set(self.A) & set(self.B)
        self.union = set(self.A) | set(self.B)

    def update(self, add: str, remove: str):
        """
        Update the Jaccard similarity between the query and the window
        where one token is added and one token is removed.
        """
        # assert remove in self.B, f"{remove} not in {self.B}"
        # assert remove in self.union, f"{remove} not in {self.union}"
        if add == remove:
            # No action is necessary
            return self.jaccard_similarity

        if add not in self.B:
            if add in self.A:
                self.intersection.add(add)
            self.union.add(add)

        self.B[add] += 1
        self.B[remove] -= 1

        if self.B[remove] == 0:
            del self.B[remove]
            if remove not in self.A:
                self.union.remove(remove)
            else:
                self.intersection.remove(remove)
        return self.jaccard_similarity

    @property
    def jaccard_similarity(self) -> float:
        intersection = len(self.intersection)
        # Avoid division by zero
        union = max(1, len(self.union))
        return intersection / union


def best_fuzzy_search(query: str, doc: str) -> Tuple[int, str]:
    """
    Find the best fuzzy match between the query and the document, assuming that
    the match shares 1) the same prefix and 2) same number of tokens. This is useful
    when you have different PDF text extraction where you might have slightly
    different word orders but same words.

    Parameters
    ----------
    query : str
        The query.
    doc : str
        The document.

    Returns
    -------
    Tuple[int, str]
        The best fuzzy match start index and the matched string.

    Examples
    --------
    >>> best_fuzzy_search("Hello world!", "Random word, Hello word! hello menudo!")
    (13, 'Hello word!')
    """
    query_tokens, _ = tokenize(query, n_gram=1)
    tokens, offsets = tokenize(doc, n_gram=1)

    matcher = BestFuzzyMatcher(query_tokens, tokens[: len(query_tokens)])

    best_score: float = matcher.jaccard_similarity
    best_start: int = 0
    best_end: int = offsets[len(query_tokens) - 1][1]

    for i in tqdm(range(len(query_tokens), len(tokens)), desc="Fuzzy searching..."):
        score = matcher.update(tokens[i], tokens[i - len(query_tokens)])
        if score >= best_score:
            best_score = score
            best_start = offsets[i - len(query_tokens) + 1][0]
            best_end = offsets[i][1]

    return best_start, doc[best_start:best_end]
