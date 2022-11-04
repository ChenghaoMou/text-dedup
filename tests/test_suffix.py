#!/usr/bin/env python
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from __future__ import annotations

from text_dedup.exact_dedup import PythonSuffixArray


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

    embedder = PythonSuffixArray(k=10, merge_strategy='longest')
    slices = embedder.fit_predict(corpus)

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

    embedder = PythonSuffixArray(k=10, merge_strategy='overlapping')
    slices = embedder.fit_predict(corpus)

    for _, intervals, results in zip(corpus, slices, targets):
        assert intervals == results, f'{intervals} != {results}'


# def test_suffix_bash():
#
#     corpus = [
#         'The quick brown fox jumps over the lazy dog',
#         'The quick brown fox jumps over the lazy dog',
#         'This is a test',
#         'This is a test',
#         'This is a random test',
#         'The quick brown fox and a random test',
#     ]
#     targets = [
#         [slice(0, 43, None)],
#         [slice(0, 43, None)],
#         [slice(0, 14, None)],
#         [slice(0, 14, None)],
#         [slice(0, 21, None)],
#         [slice(0, 20, None), slice(23, 37, None)],
#     ]
#
#     embedder = PythonSuffixArrayDeduplicator(k=10, merge=True, merge_strategy='overlapping')
#     slices = embedder.fit_predict(corpus)
#
#     for _, intervals, results in zip(corpus, slices, targets):
#         assert intervals == results
#
#
# def test_suffix_corss_bash():
#
#     corpus_a = [
#         'The quick brown fox jumps over the lazy dog',
#         'The quick brown fox jumps over the lazy dog',
#         'This is a test',
#         'This is a test',
#         'This is a random test',
#         'The quick brown fox and a random test',
#     ]
#     corpus_b = ['The quick brown fox jumps over the lazy dog',
#                 'This is a random string that would have a match.']
#     targets_a = [
#         [
#             slice(
#                 0, 43, None)], [
#             slice(
#                 0, 43, None)], [
#             slice(
#                 0, 10, None)], [
#             slice(
#                 0, 10, None)], [
#             slice(
#                 0, 17, None)], [
#             slice(
#                 0, 20, None), slice(
#                 23, 33, None)]]
#     targets_b = [[slice(0, 43, None)], [slice(0, 17, None)]]
#
#     embedder = SuffixArrayEmbedder(k=10)
#     slices_a, slices_b = embedder.cross_embed_bash(
#         corpus_a, corpus_b, merge=True, merge_strategy='overlapping', skip_existing=False)
#
#     for _, intervals, results in zip(corpus_a, slices_a, targets_a):
#         assert intervals == results
#
#     for _, intervals, results in zip(corpus_b, slices_b, targets_b):
#         assert intervals == results


# def test_two_implementations():
#
#     corpus = [
#         'The quick brown fox jumps over the lazy dog',
#         'The quick brown fox jumps over the lazy dog',
#         'This is a test',
#         'This is a test',
#         'This is a random test',
#         'The quick brown fox and a random test',
#     ]
#     targets = [
#         [slice(0, 43, None)],
#         [slice(0, 43, None)],
#         [slice(0, 14, None)],
#         [slice(0, 14, None)],
#         [slice(0, 21, None)],
#         [slice(0, 20, None), slice(23, 37, None)],
#     ]
#
#     embedder = SuffixArrayEmbedder(k=10)
#     slices_a = embedder.embed_bash(corpus, merge=True, merge_strategy='overlapping')
#     slices_b = embedder.embed(corpus, merge=True, merge_strategy='overlapping')
#
#     assert slices_a == slices_b, f'{slices_a} != {slices_b}'
#
#     for _, intervals, results in zip(corpus, slices_a, targets):
#         assert intervals == results, f'{intervals} != {results}'
