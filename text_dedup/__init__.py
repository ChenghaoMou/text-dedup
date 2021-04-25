#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-13 09:07:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Union, Optional
from text_dedup.groupers import Grouper
import pandas as pd

def drop_duplicates(df: Union[pd.DataFrame, pd.Series], deduper: Grouper, column: Optional[str] = None) -> Union[pd.DataFrame, pd.Series]:
    """Drop nearly duplicates with a deduper.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input dataframe or series
    deduper : Grouper
        Grouper
    column : Optional[str], optional
        Target column for dataframe, by default None

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Deduped data

    Examples
    --------
    >>> df = drop_duplicates(pd.DataFrame({"a": ["test", "test", "other"]}), deduper=Grouper(), column="a")
    >>> df.shape
    (2, 1)
    >>> df = drop_duplicates(pd.DataFrame({"a": ["This is a test message", "A test message", "other"]}), deduper=EditDistanceSimilarityGrouper(similarity_metric="cosine", threshold=0.7, k=1), column="a")
    >>> df.shape
    (2, 1)
    """

    df = group_duplicates(df, deduper=deduper, column=column, target_column='__group_label__')

    ans = df.groupby("__group_label__").first()

    return ans

def group_duplicates(df: Union[pd.DataFrame, pd.Series], deduper: Grouper, column: Optional[str] = None, target_column: str = "group") -> Union[pd.DataFrame, pd.Series]:
    """Group duplicates and add a new column for group label.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input dataframe or series
    deduper : Grouper
        Grouper
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
    >>> df = group_duplicates(pd.DataFrame({"a": ["test", "test", "other"]}), deduper=Grouper(), column="a")
    >>> df["group"].values.tolist()
    [0, 0, 2]
    """

    if column is not None:
        col = df[column]
    else:
        col = df
    

    group_labels = deduper.fit_transform(col)

    df[target_column] = group_labels
    return df