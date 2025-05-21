#!/usr/bin/env python
# @Date    : 2023-08-12 22:18:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import math
import re
import sys
import time
import warnings
from itertools import tee
from logging import Logger
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
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import BooleanType
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
def ngrams(sequence: List[str], n: int, min_length: int = 5):
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


def ngram_hashes(content: str, n: int, min_length: int = 5) -> Set[int]:
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
    ng: set[bytes] = {bytes(" ".join(t).lower(), "utf-8") for t in ngrams(tokens, n, min_length)}
    return {xxhash.xxh32_intdigest(n) for n in ng}


def ngrams_length_check(content: str, n: int, min_length: int = 5) -> bool:
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
        bool
        True if at least one ngram meets the `min_length` requirement, otherwise False.

    Examples
    --------
    >>> ngrams_length_check("a b c d", 2, min_length=1)
    True
    >>> ngrams_length_check("a b c d", 2, min_length=5)
    False
    >>> ngrams_length_check("a b", 3, min_length=1)
    True
    """
    tokens: List[str] = NON_ALPHA.split(content.lower())
    return len(tokens) >= min_length


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
    hashes = np.array(list(ngram_hashes(content, ngram_size, min_length)), dtype=DTYPE)
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
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory of parquet files",
    )
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum token length of document to be considered. Short ones will be removed",
    )
    parser.add_argument("--num_perm", type=int, default=250, help="Number of permutations")
    parser.add_argument("--b", type=int, default=None, help="Number of bands")
    parser.add_argument("--r", type=int, default=None, help="Number of rows per band")
    parser.add_argument("--column", "-c", type=str, default="content", help="Column to deduplicate on")
    parser.add_argument("--index", type=str, default=None, help="Column to index on")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="GCS output directory of parquet files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode by saving cluster results",
    )
    args = parser.parse_args()
    # endregion

    # region: Spark Configuration
    conf = (
        SparkConf()
        .set("spark.app.name", "MinHashLSH")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.storage.memoryFraction", "1")
        .set("spark.default.parallelism", "100")
        .set("spark.sql.autoBroadcastJoinThreshold", "20485760")
        .set("spark.sql.broadcastTimeout", "3600")
        .set("spark.sql.shuffle.partitions", "8192")
    )
    spark = SparkSession.Builder().config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(args.checkpoint_dir)
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore
    # endregion

    # region: Global Variables
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
    index_column = args.index or "__id__"

    # region: Data Loading
    # persist justification: this data will be needed when removing duplicates
    df: DataFrame = (
        spark.read.option("mergeSchema", "true")
        .parquet(args.input)
        .filter(
            udf(ngrams_length_check, BooleanType())(F.col(args.column), F.lit(args.ngram_size), F.lit(args.min_length))
        )
        .withColumn("__id__", F.monotonically_increasing_id())
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    # persist trigger
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
    for col, dtype in df.dtypes:
        log.debug(f"{col:<64}: {dtype}")
    log.debug("-" * 120)

    if DATA_SIZE == 0:
        log.debug("No data found.")
        exit(0)
    # endregion

    # region: MinHash
    edges: pyspark.RDD = (
        df.select(index_column, args.column)
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
        .groupBy(lambda x: (x[0], x[1]))  # group by (band_idx, band hash value), potential bottleneck
        .flatMap(lambda x: generate_edges([ele[2] for ele in x[1]]))
        .distinct()
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    # endregion

    # region: Connected Components

    if edges.isEmpty():
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
        df.unpersist()
        edges.unpersist()

        log.debug("-" * 120)
        log.debug("No duplicates found.")
        log.debug(f"Data Output:    {args.output}")
        log.debug(f"Time:           {time.time() - start_time:.2f}s")
        log.debug("-" * 120)

        sys.exit(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edges_df: DataFrame = (
            spark.createDataFrame(edges, schema=["src", "dst"])
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Edges DataFrame: {edges_df.count()}")
        vertices_df: DataFrame = (
            edges_df.select(F.col("src").alias("id"))
            .union(edges_df.select(F.col("dst").alias("id")))
            .distinct()
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Vertices DataFrame: {vertices_df.count()}")
        assignment: DataFrame = (
            GraphFrame(vertices_df, edges_df).connectedComponents().persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Assignment DataFrame: {assignment.count()}")
        edges_df.unpersist()
        vertices_df.unpersist()
    # endregion

    if args.debug:
        # save assignment for debugging purposes
        assignment.write.parquet(f"{args.output}-assignment/assignment.parquet", mode="overwrite")

    # region: Merge Results
    # justification: this is needed for final output
    df = df.join(
        assignment.select(F.col("id").alias(index_column), F.col("component").alias("__component__")),
        on=index_column,
        how="left",
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    assignment.unpersist()
    log.debug(f"Merging records: {df.count()}")
    # endregion

    df = (
        df.filter(F.col("__component__").isNull() | (F.col("__component__") == F.col(index_column)))
        .drop("__component__")
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    FINAL_SIZE = df.count()

    # region: Output
    partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
    df.unpersist()

    # endregion

    log.debug("-" * 120)
    log.debug(f"Number of rows before:    {DATA_SIZE}")
    log.debug(f"Number of rows after:     {FINAL_SIZE}")
    log.debug(f"Percentage of rows kept:  {FINAL_SIZE / max(0, DATA_SIZE) * 100:.2f}%")
    log.debug(f"Output:                   {args.output}")
    log.debug(f"Time:                     {time.time() - start_time:.2f}s")
    log.debug("-" * 120)
