#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/29/22
# description : Jacard similarity
from typing import List
from typing import Set


def jaccard_similarity_set(s1: Set[str], s2: Set[str]) -> float:
    """
    Calculate the jaccard similarity of two sets
    :param s1: set
    :param s2: set
    :return: float
    """
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_similarity_list(l1: List[str], l2: List[str]) -> float:
    """
    Calculate the jaccard similarity of two lists
    :param l1: list
    :param l2: list
    :return: float
    """
    return jaccard_similarity_set(set(l1), set(l2))
