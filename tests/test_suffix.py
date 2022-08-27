#!/usr/bin/env python
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

from text_dedup.embedders.suffix import SuffixArrayEmbedder


def test_suffix():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
    ]
    targets = [
        [
            slice(
                0, 43, None), slice(
                0, 20, None), slice(
                    1, 43, None), slice(
                        1, 20, None), slice(
                            2, 43, None), slice(
                                2, 20, None), slice(
                                    3, 43, None), slice(
                                        3, 20, None), slice(
                                            4, 43, None), slice(
                                                4, 20, None), slice(
                                                    5, 43, None), slice(
                                                        5, 20, None), slice(
                                                            6, 43, None), slice(
                                                                6, 20, None), slice(
                                                                    7, 43, None), slice(
                                                                        7, 20, None), slice(
                                                                            8, 43, None), slice(
                                                                                8, 20, None), slice(
                                                                                    9, 43, None), slice(
                                                                                        9, 20, None), slice(
                                                                                            10, 43, None), slice(
                                                                                                10, 20, None), slice(
                                                                                                    11, 43, None), slice(
                                                                                                        12, 43, None), slice(
                                                                                                            13, 43, None), slice(
                                                                                                                14, 43, None), slice(
                                                                                                                    15, 43, None), slice(
                                                                                                                        16, 43, None), slice(
                                                                                                                            17, 43, None), slice(
                                                                                                                                18, 43, None), slice(
                                                                                                                                    19, 43, None), slice(
                                                                                                                                        20, 43, None), slice(
                                                                                                                                            21, 43, None), slice(
                                                                                                                                                22, 43, None), slice(
                                                                                                                                                    23, 43, None), slice(
                                                                                                                                                        24, 43, None), slice(
                                                                                                                                                            25, 43, None), slice(
                                                                                                                                                                26, 43, None), slice(
                                                                                                                                                                    27, 43, None), slice(
                                                                                                                                                                        28, 43, None), slice(
                                                                                                                                                                            29, 43, None), slice(
                                                                                                                                                                                30, 43, None), slice(
                                                                                                                                                                                    31, 43, None), slice(
                                                                                                                                                                                        32, 43, None), slice(
                                                                                                                                                                                            33, 43, None)], [
                                                                                                                                                                                                slice(
                                                                                                                                                                                                    0, 43, None), slice(
                                                                                                                                                                                                        1, 43, None), slice(
                                                                                                                                                                                                            2, 43, None), slice(
                                                                                                                                                                                                                3, 43, None), slice(
                                                                                                                                                                                                                    4, 43, None), slice(
                                                                                                                                                                                                                        5, 43, None), slice(
                                                                                                                                                                                                                            6, 43, None), slice(
                                                                                                                                                                                                                                7, 43, None), slice(
                                                                                                                                                                                                                                    8, 43, None), slice(
                                                                                                                                                                                                                                        9, 43, None), slice(
                                                                                                                                                                                                                                            10, 43, None), slice(
                                                                                                                                                                                                                                                11, 43, None), slice(
                                                                                                                                                                                                                                                    12, 43, None), slice(
                                                                                                                                                                                                                                                        13, 43, None), slice(
                                                                                                                                                                                                                                                            14, 43, None), slice(
                                                                                                                                                                                                                                                                15, 43, None), slice(
                                                                                                                                                                                                                                                                    16, 43, None), slice(
                                                                                                                                                                                                                                                                        17, 43, None), slice(
                                                                                                                                                                                                                                                                            18, 43, None), slice(
                                                                                                                                                                                                                                                                                19, 43, None), slice(
                                                                                                                                                                                                                                                                                    20, 43, None), slice(
                                                                                                                                                                                                                                                                                        21, 43, None), slice(
                                                                                                                                                                                                                                                                                            22, 43, None), slice(
                                                                                                                                                                                                                                                                                                23, 43, None), slice(
                                                                                                                                                                                                                                                                                                    24, 43, None), slice(
                                                                                                                                                                                                                                                                                                        25, 43, None), slice(
                                                                                                                                                                                                                                                                                                            26, 43, None), slice(
                                                                                                                                                                                                                                                                                                                27, 43, None), slice(
                                                                                                                                                                                                                                                                                                                    28, 43, None), slice(
                                                                                                                                                                                                                                                                                                                        29, 43, None), slice(
                                                                                                                                                                                                                                                                                                                            30, 43, None), slice(
                                                                                                                                                                                                                                                                                                                                31, 43, None), slice(
                                                                                                                                                                                                                                                                                                                                    32, 43, None), slice(
                                                                                                                                                                                                                                                                                                                                        33, 43, None)], [
                                                                                                                                                                                                                                                                                                                                            slice(
                                                                                                                                                                                                                                                                                                                                                0, 14, None)], [
                                                                                                                                                                                                                                                                                                                                                    slice(
                                                                                                                                                                                                                                                                                                                                                        0, 14, None), slice(
                                                                                                                                                                                                                                                                                                                                                            0, 10, None), slice(
                                                                                                                                                                                                                                                                                                                                                                0, 10, None)], [
                                                                                                                                                                                                                                                                                                                                                                    slice(
                                                                                                                                                                                                                                                                                                                                                                        0, 10, None), slice(
                                                                                                                                                                                                                                                                                                                                                                            0, 10, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                7, 21, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                    8, 21, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                        9, 21, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                            10, 21, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                11, 21, None)], [
                                                                                                                                                                                                                                                                                                                                                                                                    slice(
                                                                                                                                                                                                                                                                                                                                                                                                        0, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                            1, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                2, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                    3, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                        4, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                            5, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                6, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                    7, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                        8, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                            9, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                10, 20, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                    23, 37, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                        24, 37, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                            25, 37, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                26, 37, None), slice(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    27, 37, None)], ]

    embedder = SuffixArrayEmbedder(k=10)
    slices = embedder.embed(corpus, merge=False)

    for _, intervals, results in zip(corpus, slices, targets):
        assert intervals == results, f'{intervals} != {results}'


def test_suffix_longest():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
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

    for _, intervals, results in zip(corpus, slices, targets):
        assert intervals == results, f'{intervals} != {results}'


def test_suffix_overlapping():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
    ]
    targets = [
        [slice(0, 43, None)],
        [slice(0, 43, None)],
        [slice(0, 14, None)],
        [slice(0, 14, None)],
        [slice(0, 21, None)],
        [slice(0, 20, None), slice(23, 37, None)],
    ]

    embedder = SuffixArrayEmbedder(k=10)
    slices = embedder.embed(corpus, merge=True, merge_strategy='overlapping')

    for _, intervals, results in zip(corpus, slices, targets):
        assert intervals == results, f'{intervals} != {results}'


def test_suffix_bash():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
    ]
    targets = [
        [slice(0, 43, None)],
        [slice(0, 43, None)],
        [slice(0, 14, None)],
        [slice(0, 14, None)],
        [slice(0, 21, None)],
        [slice(0, 20, None), slice(23, 37, None)],
    ]

    embedder = SuffixArrayEmbedder(k=10)
    slices = embedder.embed_bash(corpus, merge=True, merge_strategy='overlapping')

    for _, intervals, results in zip(corpus, slices, targets):
        assert intervals == results


def test_suffix_corss_bash():

    corpus_a = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
    ]
    corpus_b = ['The quick brown fox jumps over the lazy dog', 'This is a random string that would have a match.']
    targets_a = [
        [
            slice(
                0, 43, None)], [
            slice(
                0, 43, None)], [
            slice(
                0, 10, None)], [
            slice(
                0, 10, None)], [
            slice(
                0, 17, None)], [
            slice(
                0, 20, None), slice(
                23, 33, None)]]
    targets_b = [[slice(0, 43, None)], [slice(0, 17, None)]]

    embedder = SuffixArrayEmbedder(k=10)
    slices_a, slices_b = embedder.corss_embed_bash(
        corpus_a, corpus_b, merge=True, merge_strategy='overlapping', skip_existing=False)

    for _, intervals, results in zip(corpus_a, slices_a, targets_a):
        assert intervals == results

    for _, intervals, results in zip(corpus_b, slices_b, targets_b):
        assert intervals == results


def test_two_implementations():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
        'This is a random test',
        'The quick brown fox and a random test',
    ]
    targets = [
        [slice(0, 43, None)],
        [slice(0, 43, None)],
        [slice(0, 14, None)],
        [slice(0, 14, None)],
        [slice(0, 21, None)],
        [slice(0, 20, None), slice(23, 37, None)],
    ]

    embedder = SuffixArrayEmbedder(k=10)
    slices_a = embedder.embed_bash(corpus, merge=True, merge_strategy='overlapping')
    slices_b = embedder.embed(corpus, merge=True, merge_strategy='overlapping')

    assert slices_a == slices_b, f'{slices_a} != {slices_b}'

    for _, intervals, results in zip(corpus, slices_a, targets):
        assert intervals == results, f'{intervals} != {results}'
