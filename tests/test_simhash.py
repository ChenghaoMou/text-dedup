#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import numpy as np

from text_dedup.near_dedup import SimHashEmbedder
from text_dedup.near_dedup.simhash.simhash_embedder import compute
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


def test_compute():

    def old_way(hashes):
        counts = np.zeros(64, dtype=np.int64)
        obits = []
        for h in hashes:
            i = 0
            temp = []
            while i < 64:
                counts[i] += (h & 1) * 2 - 1
                temp.append((h & 1) * 2 - 1)
                h >>= 1
                i += 1
            obits.append(temp)

        result = 0
        for i in range(64):
            if counts[i] > 0:
                result |= 1 << i
        return result

    for _ in range(100):
        hashes = np.random.randint(0, 2**63 - 1, size=100)
        assert old_way(hashes) == compute(hashes), f'Failed {old_way(hashes)} != {compute(hashes)}'
