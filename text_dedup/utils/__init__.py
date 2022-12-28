#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:42:09
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.utils.add_args import add_bloom_filter_args
from text_dedup.utils.add_args import add_exact_hash_args
from text_dedup.utils.add_args import add_io_args
from text_dedup.utils.add_args import add_meta_args
from text_dedup.utils.add_args import add_minhash_args
from text_dedup.utils.add_args import add_sa_args
from text_dedup.utils.add_args import add_simhash_args
from text_dedup.utils.timer import Timer
from text_dedup.utils.tokenization import ngrams
from text_dedup.utils.union_find import UnionFind
