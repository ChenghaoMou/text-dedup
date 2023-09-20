#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-08-12 22:18:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import math
import re
import sys
import time
import warnings
from logging import Logger
from typing import Any
from typing import List
from typing import Set
from typing import Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    import numpy.typing as npt
    import pyspark
    import xxhash
    from graphframes import GraphFrame  # type: ignore
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
MAX_HASH = 4_294_967_295  # maximum 32-bit unsigned integer
MOD_PRIME = 4_294_967_291  # maximum 32-bit prime number


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


# region: Hashing
def ngrams(content: str, n: int, min_length: int = 5) -> Set[int]:
    """
    Return the ngrams in hash values. This function fuses few steps together for performance reasons.

    Parameters
    ----------
    content : str
        The content of the document.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    Set[int]
        The set of ngrams in hash values.

    Examples
    --------
    >>> sorted(list(ngrams("a b c d", 2, min_length=1)))
    [145323813, 433422276, 459146835]
    >>> list(ngrams("a b c d", 2, min_length=5))
    []
    >>> list(ngrams("a b", 3, min_length=1))
    [433422276]
    """
    tokens: List[str] = NON_ALPHA.split(content.lower())
    if len(tokens) < min_length:
        return set()

    ng: Set[str] = {" ".join(tokens[i : i + n]) for i in range(0, max(1, len(tokens) - n + 1))}
    return {xxhash.xxh32_intdigest(n) for n in ng}


def generate_hash_values(
    content: str,
    idx: int,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: Tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]],
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
    a, b = permutations
    hashes = np.array(list(ngrams(content, ngram_size, min_length)), dtype=DTYPE)
    p_hashes = ((np.outer(hashes, a) + b) % MOD_PRIME) & MAX_HASH
    min_hashes = np.vstack([p_hashes, np.full(num_perm, MAX_HASH, dtype=DTYPE)]).min(axis=0)
    return [(band_idx, min_hashes[start:end].data.tobytes(), idx) for band_idx, (start, end) in enumerate(hashranges)]


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
        return cluster[:1]

    cluster.sort(
        key=lambda x: (
            # license_type, the more permissive the better
            ["permissive", "no_license", "non_permissive"].index(x[-1]) if x[-1] is not None else float("inf"),
            # star_events_count, the more the better
            -x[-2] if x[-2] is not None else 0.0,
            # fork_events_count, the more the better
            -x[-3] if x[-3] is not None else 0.0,
            # visit_date, the earliest the better, tie breaker
            np.datetime64(x[-4]).astype(np.uint64) if x[-4] is not None else float("inf"),
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
    partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))

    (
        df.repartition(partitions)
        .withColumn("__pid__", F.spark_partition_id())
        .write.partitionBy("__pid__")
        .parquet(output, mode="overwrite", compression="snappy")
    )


# endregion


if __name__ == "__main__":  # pragma: no cover
    # region: Argument Parsing
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
    parser.add_argument("--output", "-o", type=str, required=True, help="GCS output directory of parquet files")
    parser.add_argument("--output_index", "-oi", type=str, help="GCS output directory of index parquet files")
    parser.add_argument("--index_only", action="store_true", help="Only output the index, skip deduplication")
    parser.add_argument("--rank", action="store_true", help="Rank the duplicates by quality indicators")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--profile_dir", type=str, default="./profile", help="Checkpoint directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    # endregion

    # region: Spark Configuration
    conf = (
        SparkConf()
        .set("spark.app.name", "MinHashLSH")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.storage.memoryFraction", "1")
        .set("spark.default.parallelism", "100")
        .set("spark.python.profile", "true" if args.profile else "false")
    )
    spark = SparkSession.Builder().config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(args.checkpoint_dir)
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore
    # endregion

    # region: Global Variables
    index_df: DataFrame | None = None
    kept_index: DataFrame | None = None
    FINAL_SIZE: int = 0
    MAX_WRITE_CHUNK_SIZE: int = 200_000
    MAX_WRITE_PARTITIONS: int = 2048

    B, R = args.b, args.r
    if B is None or R is None:
        B, R = optimal_param(args.threshold, args.num_perm)

    HASH_RANGES: List[Tuple[int, int]] = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS: Tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]] = (
        RNG.randint(1, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
        RNG.randint(0, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
    )
    # endregion

    start_time: float = time.time()

    # region: Data Loading
    df: DataFrame = (
        spark.read.option("mergeSchema", "true")
        .parquet(args.input)
        .filter(F.col("license_type") == "permissive")  # hard-coded for The Stack
        .withColumn("__id__", F.monotonically_increasing_id())
        .cache()  # justification: this data will be needed when removing low quality duplicates
    )
    DATA_SIZE: int = df.count()
    log.debug("-" * 120)
    log.debug(f"Using {B=}, {R=}")
    log.debug(f"Loaded documents: {DATA_SIZE}")
    log.debug(f"{args.input=}")
    log.debug(f"{args.output=}")
    log.debug(f"{args.threshold=}")
    log.debug(f"{args.ngram_size=}")
    log.debug(f"{args.min_length=}")
    log.debug(f"{args.num_perm=}")
    log.debug(f"{args.column=}")
    log.debug(f"{args.repo_column=}")
    for col, dtype in df.dtypes:
        log.debug(f"{col:<64}: {dtype}")
    log.debug("-" * 120)

    if DATA_SIZE == 0:
        log.debug("No data found.")
        exit(0)
    # endregion

    # region: MinHash
    buckets: pyspark.RDD = (
        df.select("__id__", args.column)
        .rdd.flatMap(
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
    ).cache()  # justification: this is needed for storing the index and clustering
    log.debug(f"Buckets: {buckets.count()}")
    # endregion

    if args.output_index:
        index_df = spark.createDataFrame(
            buckets.flatMapValues(lambda x: x), schema=["__key__", "__id__"]  # ((band_idx, hash value), idx)
        ).persist(pyspark.StorageLevel.DISK_ONLY)

    if args.output_index and args.index_only and index_df is not None:
        buckets.unpersist()
        partitioned_save(index_df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
        index_df.unpersist()
        log.debug(f"Output:                                 {args.output_index}")
        log.debug(f"Time:                                   {time.time() - start_time:.2f}s")
        sys.exit(0)

    # region: Connected Components
    # justification: this is needed for the alternating algorithm
    edges: pyspark.RDD = buckets.flatMap(lambda x: generate_edges(x[1])).distinct().cache()
    log.debug(f"Initial edges: {edges.count()}")
    buckets.unpersist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edges_df: DataFrame = spark.createDataFrame(edges, schema=["src", "dst"]).cache()
        vertices_df: DataFrame = (
            edges_df.select(F.col("src").alias("id")).union(edges_df.select(F.col("dst").alias("id"))).distinct()
        ).cache()
        assignment = GraphFrame(vertices_df, edges_df).connectedComponents().cache()
        edges_df.unpersist()
        vertices_df.unpersist()
    # endregion

    if edges.isEmpty():
        if args.output_index and index_df is not None:
            partitioned_save(index_df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
            index_df.unpersist()
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
        df.unpersist()

        log.debug("-" * 120)
        log.debug("No duplicates found.")
        log.debug(f"Data Output:    {args.output}")
        log.debug(f"Index Output:   {args.output_index}")
        log.debug(f"Time:           {time.time() - start_time:.2f}s")
        log.debug("-" * 120)

        sys.exit(0)

    # region: Merge Results
    # justification: this is needed for the merging
    df = df.join(
        assignment.select(F.col("id").alias("__id__"), F.col("component").alias("__component__")),
        on="__id__",
        how="left",
    ).cache()  # justification: this is needed for final output
    df.count()
    assignment.unpersist()
    # endregion

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
            "fork_events_count",
            "star_events_count",
            "license_type",
        ]
        if args.rank
        else [
            "__id__",
            "__component__",
            args.repo_column,
        ]
    )

    # justification: this is needed for the ranking
    cliques: pyspark.RDD = (
        (df.filter(F.col("__component__").isNotNull()).select(*rank_columns).rdd).groupBy(lambda x: x[1]).cache()
    )
    log.debug(f"Clusters: {cliques.count()}")
    # endregion

    if args.debug:
        cluster_id, _ = cliques.first()
        rows: List[Row] = df.filter(F.col("__component__") == cluster_id).head(5)
        log.debug("=" * 120)
        for i, record in enumerate(rows):
            content = "\n".join(record.content.split("\n")[:10])
            log.debug(f"{i}-th example repo name: {getattr(record, args.repo_column)}")
            log.debug(f"{i}-th example code:\n{content[:200]}\n")
            log.debug("=" * 120)

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
    ).cache()
    df.count()
    cliques.unpersist()
    df = df.filter(F.col("__component__").isNull() | F.col("__keep__"))
    if args.output_index and index_df is not None:
        kept_index = index_df.join(df.select("__id__"), on="__id__", how="inner").persist(
            pyspark.StorageLevel.DISK_ONLY
        )
    df = df.drop("__component__", "__keep__").cache()
    FINAL_SIZE = df.count()
    # endregion

    if args.output_index and kept_index is not None:
        partitioned_save(kept_index, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output_index)
        kept_index.unpersist()
    partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
    df.unpersist()

    log.debug("-" * 120)
    log.debug(f"Number of rows before:    {DATA_SIZE}")
    log.debug(f"Number of rows after:     {FINAL_SIZE}")
    log.debug(f"Percentage of rows kept:  {FINAL_SIZE / max(0, DATA_SIZE) * 100:.2f}%")
    log.debug(f"Output:                   {args.output}")
    log.debug(f"Index Output:             {args.output_index}")
    log.debug(f"Time:                     {time.time() - start_time:.2f}s")
    log.debug("-" * 120)

    if args.profile:
        sc.dump_profiles(args.profile_dir)
