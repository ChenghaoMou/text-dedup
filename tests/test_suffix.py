#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.embedders.suffix import SuffixArrayEmbedder


def test_suffix():

    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "This is a test",
        "This is a test",
        "This is a random test",
        "The quick brown fox and a random test"
    ]
    targets = [
        [slice(0, 43, None)],
        [slice(0, 43, None)],
        [slice(0, 14, None)],
        [slice(0, 14, None)],
        [slice(0, 10, None), slice(7, 21, None)],
        [slice(0, 20, None), slice(23, 37, None)],
    ]


    embedder = SuffixArrayEmbedder(k=10)
    slices = embedder.embed(corpus, merge=True, merge_strategy='longest')

    for sentence, intervals, results in zip(corpus, slices, targets):
        assert intervals == results