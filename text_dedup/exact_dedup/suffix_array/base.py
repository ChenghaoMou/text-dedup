from abc import ABC
from dataclasses import dataclass
from typing import Literal

from text_dedup.base import DuplicateFinder


# https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class SuffixArray(DuplicateFinder, ABC):
    """
    Base class for all suffix array based deduplicators.

    Parameters
    ----------
    k: int
        The minimum length in bytes of a duplicate substring.
    merge_strategy: Literal["longest", "overlapping"]
        The strategy to merge duplicate substrings, default "longest".
    """
    k: int
    merge_strategy: Literal['longest', 'overlapping'] = 'longest'
