#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List

from text_dedup.postprocess import annoy_clustering
from text_dedup.postprocess import get_group_indices
from text_dedup.semantic_dedup import TransformerEmbedder


def test_transformer():

    corpus = [
        'The quick brown fox jumps over the dog',
        'The quick brown fox jumps over the corgi',
        'This is a test',
        'This is a test message',
    ]

    embedder = TransformerEmbedder('bert-base-uncased')
    embeddings = embedder.embed(corpus)

    clusters: List[List[int]] = annoy_clustering(embeddings, f=768)
    groups = get_group_indices(clusters)
    assert groups == [0, 0, 2, 2]
