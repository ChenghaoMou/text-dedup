#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 10:50:47
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):

        self.model = SentenceTransformer(model_name)

    def embed(self, corpus: List[str], **kwargs) -> List[np.ndarray]:

        return self.model.encode(corpus, **kwargs)
