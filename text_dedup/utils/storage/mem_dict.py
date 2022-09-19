#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-06-18 09:37:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from text_dedup.utils.storage.base import StorageDict


class MemDict(StorageDict):
    """
    A dictionary-like object backed by memory.

    Parameters
    ----------
    storage_config : Dict[str, Any]
        Storage configuration.

    Examples
    --------
    >>> from text_dedup.utils.storage import MemDict
    >>> d = MemDict()
    >>> d.clear()
    >>> d.add((123, 234), (0, 213123124))
    >>> d.add((123, 234), (1, 213123124))
    >>> sorted(d[(123, 234)])
    [(0, 213123124), (1, 213123124)]
    """

    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        self.storage_config = storage_config
        self.mem: Dict[Any, List[Any]] = defaultdict(list)

    def add(self, key: str | Tuple[int, int], value: str | Tuple[int, int]):
        self.mem[key].append(value)

    def __len__(self):

        return len(self.mem)

    def __iter__(self):
        for key in self.mem:
            yield key

    def __getitem__(self, key: str | Tuple[int, int]) -> List[Tuple[int, ...]]:
        return self.mem[key]

    def clear(self):
        for k in self:
            del self.mem[k]
