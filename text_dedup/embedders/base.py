#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-08-30 20:02:00
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

Fingerprint = Union[int, str, List[slice], np.ndarray]


class Embedder:
    @abstractmethod
    def embed(self, corpus: List[str], **kwargs) -> List[Fingerprint]:
        raise NotImplementedError

    @abstractmethod
    def embed_function(self, **kwargs) -> Callable[[str], Fingerprint]:
        raise NotImplementedError
