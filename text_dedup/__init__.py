#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 12:48:33
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

from typing import List
from text_dedup.embed.stfm import SentenceTransformerEmbedder
from text_dedup.utils.nn import annoy_clustering
from text_dedup.utils.group import get_group_indices


class SentenceTransformerDeduper:
    def __init__(self, model_name: str):

        self.model = SentenceTransformerEmbedder(model_name)

    def group(self, corpus: List[str], **kwargs) -> List[int]:

        embeddings = self.model.embed(corpus, **kwargs)
        clusters = annoy_clustering(embeddings, f=embeddings[0].shape[0])

        return get_group_indices(clusters)
