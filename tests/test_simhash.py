#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.near_dedup import SimHashEmbedder
from text_dedup.postprocess import get_group_indices
from text_dedup.postprocess import simhash_clustering


def test_simhash():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
    ]

    embedder = SimHashEmbedder()
    embeddings = embedder.embed(corpus)

    clusters = simhash_clustering(embeddings)
    groups = get_group_indices(clusters)
    assert groups == [0, 0, 2, 2]
