#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 10:54:44
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

from typing import List

from text_dedup.utils.union_find import UF


def get_group_indices(neighbors: List[List[int]]) -> List[int]:
    """Based on the nearest neighbors, find the group/cluster index for each element.

    Parameters
    ----------
    neighbors : List[List[int]]
        List of nearest neighbor indices

    Returns
    -------
    List[int]
        List of group indices

    Examples
    --------
    >>> get_group_indices([[0, 1], [0, 2], [1, 2]])
    [0, 0, 0]
    """
    finder = UF(len(neighbors))
    for i, n in enumerate(neighbors):
        for j in n:
            finder.union(i, j)

    return [finder.find(i) for i in range(len(neighbors))]
