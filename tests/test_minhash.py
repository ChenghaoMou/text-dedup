#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.utils.group import get_group_indices
from text_dedup.utils.nn import lsh_clustering


def test_minhash():

    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "This is a test",
        "This is a test",
    ]

    embedder = MinHashEmbedder()
    embeddings = embedder.embed(corpus)

    clusters = lsh_clustering(embeddings)
    groups = get_group_indices(clusters)
    assert groups == [0, 0, 2, 2]