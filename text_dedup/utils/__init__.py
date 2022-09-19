#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 14:52:53
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Utility functions."""
from text_dedup.utils.hf_datasets import dataset_get
from text_dedup.utils.hf_datasets import dataset_get_all_str_columns
from text_dedup.utils.hf_datasets import dataset_map
from text_dedup.utils.hf_datasets import extract_text
from text_dedup.utils.hf_datasets import get_byte_size
from text_dedup.utils.storage import StorageDict
from text_dedup.utils.storage import create_storage

__all__ = [
    "create_storage",
    "dataset_get",
    "dataset_get_all_str_columns",
    "dataset_map",
    "extract_text",
    "get_byte_size",
    "StorageDict",
]
