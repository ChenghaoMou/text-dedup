#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-08-12 22:18:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import hashlib
import math
import os
import re
import struct
import sys
import time
from itertools import tee
from logging import Logger
from operator import add
from typing import Any
from typing import List
from typing import Text
from typing import Tuple

import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkConf
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.integrate import quad as integrate

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
DTYPE = np.uint32
MAX_HASH = 4_294_967_295
MOD_PRIME = 4_294_967_291


# region: Connected Components in MapReduce and Beyond, 2014
def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.

    Examples
    --------
    >>> generate_edges([1, 2, 3])
    [(2, 1), (3, 1)]
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


def small_star(edges):
    def small_star_map(edge):
        x, y = edge
        if y <= x:
            return (x, y)
        else:
            return (y, x)

    def small_star_reduce(x):
        node, neighbors = x
        nodes = neighbors + [node]
        min_node = min(nodes)
        new_edges = set((neighbor, min_node) for neighbor in nodes if (neighbor <= node and neighbor != min_node))
        change = len(new_edges.difference(set([(node, neighbor) for neighbor in neighbors])))
        return (list(new_edges), change)

    neighbors = edges.map(small_star_map).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    edges_with_change = neighbors.map(small_star_reduce).cache()
    if edges_with_change.isEmpty():
        total_change = 0
    else:
        total_change = edges_with_change.map(lambda x: x[1]).reduce(add)
    edges = edges_with_change.flatMap(lambda x: x[0])
    edges_with_change.unpersist()

    return edges, total_change


def large_star(edges):
    def large_star_map(edge):
        if edge[0] == edge[1]:
            return [(edge[0], edge[1])]
        return [(edge[0], edge[1]), (edge[1], edge[0])]

    def large_star_reduce(x):
        node, neighbors = x
        nodes = neighbors + [node]
        min_node = min(nodes)
        new_edges = set((neighbor, min_node) for neighbor in (neighbors + [node]) if (neighbor > node))
        change = len(new_edges.difference(set([(node, neighbor) for neighbor in neighbors])))
        return list(new_edges), change

    neighbors = edges.flatMap(large_star_map).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    edges_with_change = neighbors.map(large_star_reduce).cache()
    if edges_with_change.isEmpty():
        total_change = 0
    else:
        total_change = edges_with_change.map(lambda x: x[1]).reduce(add)
    edges = edges_with_change.flatMap(lambda x: x[0])
    edges_with_change.unpersist()
    return edges, total_change


def alternating_algo(edges, max_iteration: int) -> Tuple[Any, bool, int]:

    prev_lchanges: int = sys.maxsize
    prev_schanges: int = sys.maxsize
    curr_iteration: int = 0

    while max_iteration:

        edges, curr_lchanges = large_star(edges)
        edges, curr_schanges = small_star(edges)

        if (curr_lchanges == prev_lchanges and curr_schanges == prev_schanges) or (
            curr_schanges == 0 and curr_lchanges == 0
        ):
            return edges, True, curr_iteration

        prev_lchanges = curr_lchanges
        prev_schanges = curr_schanges
        curr_iteration += 1
        max_iteration -= 1

    return edges, False, curr_iteration


# endregion

# region: Hashing
def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=1))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=5))
    []
    >>> list(ngrams(["a", "b"], 3, min_length=1))
    [('a', 'b')]
    """
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid abstraction.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
        The first 4 bytes (32 bits) of the SHA1 hash of the input data.

    Examples
    --------
    >>> sha1_hash32(b"hello")
    499578026
    >>> bin(sha1_hash32(b"hello"))
    '0b11101110001101111010010101010'
    >>> sha1_hash32(b"hello world").bit_length()
    30
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def generate_hash_values(
    content: str,
    idx: int,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[int, bytes, int]]:
    """
    Generate the MinHashLSH values for a given document.

    Parameters
    ----------
    content : str
        The content of the document.
    idx : int
        The index of the document.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of the n-grams.
    min_length : int
        The minimum number of tokens in a document.
    hashranges : list
        The ranges of offsets for each hash value.
    permutations : Tuple[np.ndarray, np.ndarray]
        The permutations for the hash values.

    Returns
    -------
    List[Tuple[int, bytes, int]]
        The list of (band_idx, hash value, idx) for the document.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> PERMUTATIONS = (
    ...     RNG.randint(1, MOD_PRIME, size=(num_perm,), dtype=DTYPE),
    ...     RNG.randint(0, MOD_PRIME, size=(num_perm,), dtype=DTYPE),
    ... )
    >>> res = generate_hash_values(content, idx, num_perm, ngram_size, 0, hashranges, PERMUTATIONS)
    >>> len(res)
    10
    >>> sum(len(h) for _, h, _ in res) == len(res) * 25 * np.dtype(DTYPE).itemsize
    True
    """
    tokens = {" ".join(t).encode("utf-8") for t in ngrams(NON_ALPHA.split(content.lower()), ngram_size, min_length)}
    a, b = permutations
    hv = np.array([sha1_hash32(token) for token in tokens], dtype=DTYPE)
    phv = np.bitwise_and(((hv * np.tile(a, (len(tokens), 1)).T).T + b) % MOD_PRIME, MAX_HASH)  # type: ignore
    hash_values = np.vstack([phv, np.ones(num_perm, dtype=DTYPE) * MAX_HASH]).min(axis=0)
    Hs = [bytes(hash_values[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]


# endregion

# region: MinHashLSH
def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(area, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(area, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


# endregion

# region: Quality Control
def process_cluster(cluster: List[Any], enabled: bool = False) -> List[Any]:
    if not enabled:
        np.random.shuffle(cluster)
        return cluster[:1]

    cluster.sort(
        key=lambda x: (
            -x[-1] if x[-1] is not None else 0.0,  # star_events_count
            -x[-2] if x[-2] is not None else 0.0,  # fork_events_count
            -np.datetime64(x[-3]).astype(np.uint64) if x[-3] is not None else 0.0,  # visit_date
        )
    )
    return cluster[:1]


# endregion

# region: IO
def partitioned_save(df: DataFrame, chunk_size: int, max_partitions: int, output: str):
    """
    Save a Spark DataFrame to a GCS directory in batches of `chunk_size` rows. PySpark natively does not support this
    functionality, so this workaround is necessary.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The Spark DataFrame to save.
    chunk_size : int
        The number of rows per batch.
    max_partitions : int
        The maximum number of partitions.
    output : str
        The GCS output directory.

    Raises
    ------
    RuntimeError
        If the save fails.
    """

    total_rows = df.count()
    partitions = max(1, min(math.ceil(total_rows / chunk_size), max_partitions))

    def save_partition(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        pid = df["__pid__"].iloc[0]
        df = df.drop("__pid__", axis=1)
        df.to_parquet(
            os.path.join(output, f"part-{pid:05d}-{partitions:05d}.parquet"), index=False, compression="snappy"
        )
        return pd.DataFrame([{"__status__": True, "__pid__": pid}])

    results = (
        df.repartition(partitions)  # random and uniform hash partitioning
        .withColumn("__pid__", F.spark_partition_id())
        .groupBy("__pid__")
        .applyInPandas(save_partition, schema="__status__ boolean, __pid__ int")
        .toPandas()
    )

    if results["__status__"].all():
        pd.DataFrame([]).to_csv(os.path.join(output, "_SUCCESS"), index=False, header=False)
        return

    raise RuntimeError("Failed to save partitions.")


# endregion


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(description="Intra-dataset near-deduplicating with PySpark")
    parser.add_argument("--input", "-i", type=str, required=True, help="GCS input directory of parquet files")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum token length of document to be considered")
    parser.add_argument("--num_perm", type=int, default=250, help="Number of permutations")
    parser.add_argument("--b", type=int, default=None, help="Number of bands")
    parser.add_argument("--r", type=int, default=None, help="Number of rows per band")
    parser.add_argument("--column", "-c", type=str, default="content", help="Column to deduplicate on")
    parser.add_argument("--repo_column", type=str, required=True, help="Code repo column")
    parser.add_argument("--index_column", type=str, default=None, help="Index column, will be assigned if not provided")
    parser.add_argument("--output", "-o", type=str, required=True, help="GCS output directory of parquet files")
    parser.add_argument("--output_index", "-oi", type=str, help="GCS output directory of index parquet files")
    parser.add_argument("--index_only", action="store_true", help="Only output the index, skip deduplication")
    parser.add_argument("--rank", action="store_true", help="Rank the duplicates by quality indicators")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    conf = SparkConf()
    conf.set("spark.app.name", "MinHashLSH")
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()  # type: ignore
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore

    index_df: DataFrame = None  # type: ignore
    kept_index: DataFrame = None  # type: ignore

    start_time = time.time()

    B, R = args.b, args.r
    if B is None or R is None:
        B, R = optimal_param(args.threshold, args.num_perm)

    MAX_WRITE_CHUNK_SIZE: int = 1_000_000
    MAX_WRITE_PARTITIONS: int = 256
    HASH_RANGES: List[Tuple[int, int]] = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS: Tuple[np.ndarray, np.ndarray] = (
        RNG.randint(1, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
        RNG.randint(0, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
    )

    # region: Data Loading
    df: DataFrame = spark.read.option("mergeSchema", "true").parquet(args.input)
    if args.index_column is None:
        df = df.withColumn("__id__", F.monotonically_increasing_id()).cache()
    else:
        df = df.withColumn("__id__", F.col(args.index_column)).cache()

    DATA_SIZE: int = df.count()
    # endregion

    if args.debug:

        log.debug("-" * 120)
        log.debug(f"Using {B=}, {R=}")
        log.debug(f"{args.input=}")
        log.debug(f"{args.output=}")
        log.debug(f"{args.threshold=}")
        log.debug(f"{args.ngram_size=}")
        log.debug(f"{args.min_length=}")
        log.debug(f"{args.num_perm=}")
        log.debug(f"{args.column=}")
        log.debug(f"{args.repo_column=}")
        log.debug(f"{args.index_column=}")
        for col, dtype in df.dtypes:
            log.debug(f"{col:<64}: {dtype}")
        log.debug("-" * 120)

    # region: MinHash
    records: pyspark.RDD = df.select("__id__", args.column).rdd.cache()
    buckets: pyspark.RDD = (
        records.flatMap(
            lambda x: generate_hash_values(
                content=x[1],  # args.column
                idx=x[0],  # __id__
                num_perm=args.num_perm,
                ngram_size=args.ngram_size,
                min_length=args.min_length,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
            )
        )  # (band_idx, band hash value, idx)
        .groupBy(lambda x: (x[0], x[1]))  # group by (band_idx, band hash value)
        .mapValues(lambda x: [ele[2] for ele in x])  # ((band_idx, hash value), [idx, ...])
    ).cache()
    records.unpersist()
    # endregion

    if args.output_index:
        index_df = spark.createDataFrame(
            buckets.flatMapValues(lambda x: x), schema=["__key__", "__id__"]  # ((band_idx, hash value), idx)
        ).persist(pyspark.StorageLevel.DISK_ONLY)

    if args.output_index and args.index_only and index_df is not None:
        partitioned_save(index_df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
        log.debug(f"Output:                                 {args.output_index}")
        log.debug(f"Time:                                   {time.time() - start_time:.2f}s")
        sys.exit(0)

    # region: Connected Components
    edges: pyspark.RDD = buckets.flatMap(lambda x: generate_edges(x[1])).distinct().cache()
    buckets.unpersist()
    duplicate_edges, converged, iteration = alternating_algo(edges, max_iteration=20)
    duplicate_edges.cache()
    edges.unpersist()
    # endregion

    if duplicate_edges.isEmpty():
        if args.output_index and index_df is not None:
            partitioned_save(index_df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)

        log.debug("-" * 120)
        log.debug("No duplicates found.")
        log.debug(f"Data Output:    {args.output}")
        log.debug(f"Index Output:   {args.output_index}")
        log.debug(f"Time:           {time.time() - start_time:.2f}s")
        log.debug("-" * 120)

        sys.exit(0)

    # region: Merge Results
    self_edges: pyspark.RDD = duplicate_edges.values().distinct().map(lambda x: (x, x)).cache()
    df = df.join(
        spark.createDataFrame(duplicate_edges.union(self_edges), schema=["__id__", "__component__"]),
        on="__id__",
        how="left",
    ).cache()
    duplicate_edges.unpersist()
    # endregion

    if args.debug:
        NUM_CLUSTER = self_edges.count()

    self_edges.unpersist()

    # region: Quality Control: This section is hard-coded for The Stack
    #
    # A repo's quality is measured by, in order of importance:
    #  1. The number of stars (higher is better)
    #  2. The number of forks (higher is better)
    #
    # A file's quality is therefore measured by the quality of its repo to prioritize
    # the integrity of the repo so training context can be maximized at the repo level.
    # directory_id                     object
    # blob_id                          object
    # content_id                       object
    # path                             object
    # length                            int64
    # content                          object
    # src_encoding                     object
    # language                         object
    # is_vendor                          bool
    # is_generated                       bool
    # blob_prefix                      object
    # repo_name                        object
    # repo_url                         object
    # snapshot_id                      object
    # revision_id                      object
    # branch_name                      object
    # visit_date               datetime64[ns]
    # revision_date            datetime64[ns]
    # committer_date           datetime64[ns]
    # github_id                       float64
    # star_events_count                 int64
    # fork_events_count                 int64
    # gha_license_id                   object
    # gha_fork                         object
    # gha_event_created_at     datetime64[ns]
    # gha_created_at           datetime64[ns]
    # gha_updated_at           datetime64[ns]
    # gha_pushed_at            datetime64[ns]
    # gha_size                        float64
    # gha_stargazers_count            float64
    # gha_forks_count                 float64
    # gha_open_issues_count           float64
    # gha_language                     object
    # gha_archived                     object
    # gha_disabled                     object
    # detected_licenses                object
    # license_type                     object

    rank_columns = (
        [
            "__id__",
            "__component__",
            args.repo_column,
            "visit_date",
            "star_events_count",
            "fork_events_count"
            # "max_stars_repo_stars_event_min_datetime",
            # "max_stars_count",
            # "max_forks_count",
        ]
        if args.rank
        else [
            "__id__",
            "__component__",
            args.repo_column,
        ]
    )

    duplicates: pyspark.RDD = (df.filter(F.col("__component__").isNotNull()).select(*rank_columns).rdd).cache()

    if args.debug:
        NUM_DUPLICATE = duplicates.count()
    cliques: pyspark.RDD = duplicates.groupBy(lambda x: x[1]).cache()
    duplicates.unpersist()

    # endregion

    if args.debug:
        cluster_id, _ = cliques.first()
        rows: List[Row] = df.filter(F.col("__component__") == cluster_id).head(5)
        log.debug("-" * 120)
        for i, record in enumerate(rows):
            content = "\n".join(record.content.split("\n")[:10])
            log.debug(f"{i}-th example repo name: {getattr(record, args.repo_column)}")
            log.debug(f"{i}-th example code:\n{content[:200]}\n")
            log.debug("-" * 120)

    # region: Remove Low Quality Duplicates
    df = df.join(
        spark.createDataFrame(
            cliques.mapValues(lambda x: process_cluster(cluster=list(x), enabled=args.rank)).flatMap(
                lambda x: [(ele[0], True) for ele in x[1]]
            ),
            schema=["__id__", "__keep__"],
        ),
        on="__id__",
        how="left",
    )
    cliques.unpersist()
    df = df.filter(F.col("__component__").isNull() | F.col("__keep__")).cache()
    if args.output_index and index_df is not None:
        kept_index = index_df.join(df.select("__id__"), on="__id__", how="inner").persist(
            pyspark.StorageLevel.DISK_ONLY
        )
    df = df.drop("__component__", "__keep__").cache()
    # endregion

    FINAL_SIZE = df.count()
    if args.output_index and kept_index is not None:
        partitioned_save(kept_index, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
    partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)

    if args.debug:
        log.debug(f"CC converged:                  {converged}")
        log.debug(f"CC iterations:                 {iteration}")
        log.debug(f"Number of rows before:         {DATA_SIZE}")  # type: ignore
        log.debug(f"Number of duplicate rows:      {NUM_DUPLICATE}")  # type: ignore
        log.debug(f"Number of duplicate clusters:  {NUM_CLUSTER}")  # type: ignore
        log.debug(f"Number of rows after:          {FINAL_SIZE}")
        log.debug(f"Percentage of rows kept:       {FINAL_SIZE / max(0, DATA_SIZE) * 100:.2f}%")  # type: ignore

    log.debug("-" * 120)
    log.debug(f"Output:        {args.output}")
    log.debug(f"Index Output:  {args.output_index}")
    log.debug(f"Time:          {time.time() - start_time:.2f}s")
    log.debug("-" * 120)
