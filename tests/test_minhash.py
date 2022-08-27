#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import time

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.postprocess.clustering import lsh_clustering
from text_dedup.postprocess.group import get_group_indices


def test_minhash():

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',
        'This is a test',
        'This is a test',
    ]

    embedder = MinHashEmbedder()
    embeddings = embedder.embed(corpus)

    clusters = lsh_clustering(embeddings)
    groups = get_group_indices(clusters)
    assert groups == [0, 0, 2, 2]


def __minhash_num_perm(num_perm: int = 128):

    doc = '''
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque sed justo nec elit condimentum placerat a quis ipsum. Donec faucibus sem id ipsum laoreet suscipit. Sed ut semper lacus, a hendrerit massa. Duis imperdiet risus massa, ut tempus quam lobortis eget. Aliquam erat volutpat. Suspendisse lobortis et felis id luctus. Suspendisse in purus nec lacus elementum aliquam eu at metus. Curabitur et consectetur tortor. Nullam ut orci et erat tincidunt rutrum. Proin porta scelerisque tempus.

        Cras vel blandit sapien. Vestibulum condimentum id ex vitae rutrum. Nulla ac massa lacinia, feugiat quam ac, ornare risus. Integer nec hendrerit nisl. Proin nisl ante, viverra quis lobortis vel, pretium ut massa. Nullam laoreet lacus ex, quis varius mi sagittis ut. Etiam ac ante nibh. Morbi congue arcu interdum arcu feugiat, dictum pulvinar dolor dignissim. Interdum et malesuada fames ac ante ipsum primis in faucibus.

        Donec lobortis pharetra ex, lacinia sagittis est mollis ac. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. In pretium turpis lobortis auctor condimentum. Fusce aliquam convallis orci, vel porttitor diam luctus eget. Sed eu ipsum eget odio sollicitudin hendrerit nec vestibulum orci. Mauris vitae porta nisl. Sed congue eleifend mauris, ac interdum metus luctus in. Praesent pretium quis quam quis dictum. Maecenas ultrices quis est nec pulvinar. Donec at ultricies libero. Proin feugiat eu magna ac gravida. Proin nec convallis quam.

        Pellentesque ut arcu id turpis vehicula dictum viverra sit amet nunc. Nunc tempus diam in lorem accumsan molestie. Vestibulum blandit enim a felis aliquet, vel vehicula ante tempor. Praesent molestie lacus eget tellus laoreet fermentum nec vel ipsum. Etiam at posuere elit, vulputate placerat lacus. Nulla facilisi. Quisque finibus placerat nibh, quis tempor nisi posuere vel. Aenean dignissim, sem quis scelerisque euismod, magna mi scelerisque lorem, nec cursus est ante id leo. Pellentesque maximus eu ex eget pretium.

        Nam vel dolor diam. Pellentesque ornare, tortor in sollicitudin vestibulum, dui est blandit lacus, fermentum tempor turpis magna ut lorem. Nunc eget nibh laoreet, porta quam eget, fermentum orci. Vestibulum sapien turpis, volutpat id semper eu, imperdiet id mauris. Sed ac imperdiet metus. Vestibulum ac lorem justo. Donec dapibus sem nunc, et dapibus risus molestie sit amet.
    '''
    embedder = MinHashEmbedder(num_perm=num_perm)
    start_time = time.time()
    _ = embedder.embed([doc for _ in range(100)])
    end_time = time.time()

    return (end_time - start_time) / 100


def test_minhash_speed(benchmark):

    results = benchmark(__minhash_num_perm, num_perm=128)
    assert results <= 0.01, 'MinHashEmbedder is too slow'
