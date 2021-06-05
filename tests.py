#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 11:09:16
# @Author       : Chenghao Mou (mouchenghao@gmail.com)


def test_performance(fraction: float = 0.05):

    from datasets import load_dataset
    from text_dedup import SentenceTransformerDeduper
    from sklearn.metrics import f1_score, classification_report
    from alive_progress import alive_bar

    dataset = load_dataset("quora")["train"]

    corpus = {}
    pairs = []
    with alive_bar() as bar:
        for i in range(int(len(dataset) * fraction)):
            row = dataset[i]
            for text in row["questions"]["text"]:
                corpus[len(corpus)] = text
            pairs.append((len(corpus) - 1, len(corpus) - 2, row["is_duplicate"]))
            bar()

    deduper = SentenceTransformerDeduper("distilbert-base-nli-stsb-mean-tokens")
    indices = deduper.group(list(corpus.values()), show_progress_bar=True)

    predictions = []
    labels = []

    for x, y, label in pairs:
        predictions.append(indices[x] == indices[y])
        labels.append(label)

    result = f1_score(labels, predictions)
    assert result >= 0.5
    print(classification_report(labels, predictions))
    return result


def test_scaling(benchmark):

    result = benchmark.pedantic(test_performance, args=(0.05,), iterations=10)  # 16k

    assert result >= 0.5
