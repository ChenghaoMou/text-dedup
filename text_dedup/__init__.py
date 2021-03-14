#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-13 09:07:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Union, Optional
from text_dedup.dedupers import Deduper, EditDistanceSimilarityDeduper, PretrainedWordEmbeddingDeduper
from collections import deque
from alive_progress import alive_bar
import pandas as pd

def drop_duplicates(df: Union[pd.DataFrame, pd.Series], deduper: Deduper, column: Optional[str] = None) -> Union[pd.DataFrame, pd.Series]:
    """Drop nearly duplicates with a deduper.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input dataframe or series
    deduper : Deduper
        Deduper
    column : Optional[str], optional
        Target column for dataframe, by default None

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Deduped data

    Examples
    --------
    >>> df = drop_duplicates(pd.DataFrame({"a": ["test", "test", "other"]}), deduper=Deduper(), column="a")
    >>> df.shape
    (2, 1)
    >>> df = drop_duplicates(pd.DataFrame({"a": ["This is a test message", "A test message", "other"]}), deduper=EditDistanceSimilarityDeduper(similarity_metric="cosine", threshold=0.7, k=1), column="a")
    >>> df.shape
    (2, 1)
    """

    df = group_duplicates(df, deduper=deduper, column=column, target_column='__group_label__')

    ans = df.groupby("__group_label__").first()

    return ans

def group_duplicates(df: Union[pd.DataFrame, pd.Series], deduper: Deduper, column: Optional[str] = None, target_column: str = "group") -> Union[pd.DataFrame, pd.Series]:
    """Group duplicates and add a new column for group label.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input dataframe or series
    deduper : Deduper
        Deduper
    column : Optional[str], optional
        Target column for dataframe, by default None
    target_column: str
        Where to put the group label

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Group labeled data

    Examples
    --------
    >>> df = group_duplicates(pd.DataFrame({"a": ["test", "test", "other"]}), deduper=Deduper(), column="a")
    >>> df["group"].values.tolist()
    [0, 0, 2]
    """

    if column is not None:
        col = df[column]
    else:
        col = df
    
    # Construct similarity matrix
    duplicates = []

    # with alive_bar(len(col)) as bar: 
    for i, x in enumerate(col):
        duplicates.append([duplicates[j][i] if j < i else deduper.compare(x, y) if x != y else True for j, y in enumerate(col)])
            # bar()

    h = len(duplicates)
    w = len(duplicates[0]) if h else 0
    parent = {}

    for i in range(h):
        parent[i] = parent.get(i, i)
        for j in range(w):
            if duplicates[i][j] is True:
                if j not in parent:
                    parent[j] = j
                parent[i] = min(parent[i], parent[j])
    
    df[target_column] = [parent[i] for i in range(h)]

    return df