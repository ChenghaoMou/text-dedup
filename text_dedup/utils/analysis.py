#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-01-02 15:18:55
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import pickle
from collections import defaultdict

import datasets
import numpy as np
from tqdm import tqdm

from text_dedup.utils.tokenization import ngrams
from text_dedup.utils.union_find import UnionFind


def jaccard_similarity(doc1, doc2):

    words1 = set(" ".join(ng) for ng in ngrams(doc1, 8))
    words2 = set(" ".join(ng) for ng in ngrams(doc2, 8))
    return len(words1 & words2) / max(1, len(words1 | words2))


def false_positives(
    ds: datasets.Dataset,
    uf: UnionFind,
    threshold: float,
):

    groups = defaultdict(set)
    for x, y in uf.parent.items():
        groups[y].add(x)

    clusters = [c for c in groups.values() if len(c) > 1]
    false_positive_total = 0
    total = 0
    for cluster in tqdm(clusters):
        total += len(cluster)
        similarities = np.zeros((len(cluster), len(cluster)))
        for i, x in enumerate(cluster):
            for j in range(i + 1, len(cluster)):
                y = cluster[j]
                similarities[i, j] = similarities[j, i] = jaccard_similarity(ds[x]["text"], ds[y]["text"])

        false_positive_cnt = (np.max(similarities, axis=1) < threshold).sum()
        false_positive_total += false_positive_cnt

    return false_positive_total / total


if __name__ == "__main__":
    ds = datasets.load_dataset("wiki40b", "en", split="train")
    with open("output/simhash/dedup/uf.pkl", "rb") as f:
        uf = pickle.load(f)
    print(false_positives(ds, uf, 0.7))
