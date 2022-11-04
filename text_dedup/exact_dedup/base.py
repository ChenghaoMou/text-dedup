#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/1/22
# description : Base classes for exact deduplication.

from typing import List
from typing import Sequence

from text_dedup.base import DuplicateFinder


class ExactDuplicateFinder(DuplicateFinder):

    def fit(self, data: Sequence[str]):
        raise NotImplementedError

    def predict(self, data: Sequence[str]) -> List[bool]:
        raise NotImplementedError

    def fit_predict(self, data: Sequence[str]) -> List[bool]:
        raise NotImplementedError
