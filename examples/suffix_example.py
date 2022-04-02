#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.embedders.suffix import SuffixArrayEmbedder

if __name__ == "__main__":

    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "This is a test",
        "This is a test",
        "This is a random test",
        "The quick brown fox and a random test"
    ]


    embedder = SuffixArrayEmbedder(k=10)
    slices = embedder.embed(corpus, merge=True, merge_strategy='longest')

    for sentence, intervals in zip(corpus, slices):
        print(sentence)
        print([sentence[slice] for slice in intervals])