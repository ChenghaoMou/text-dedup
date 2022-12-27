#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:37:44
# @Author  : Chenghao Mou (mouchenghao@gmail.com)


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)
