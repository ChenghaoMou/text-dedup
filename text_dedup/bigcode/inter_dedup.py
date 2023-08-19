#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-08-12 22:18:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import math
import time
from logging import Logger
from typing import Set

from pyspark import SparkConf
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def remove_rows(df: DataFrame, ids: DataFrame) -> DataFrame:
    """
    Removed rows from df that have __id__ in ids.

    Parameters
    ----------
    df : DataFrame
        DataFrame to remove rows from.
    ids : DataFrame
        DataFrame of __id__ to remove.

    Returns
    -------
    DataFrame
        DataFrame with rows removed.
    """
    ids = ids.withColumn("__remove__", F.lit(True))
    return df.join(ids, on="__id__", how="left").filter(F.col("__remove__").isNull()).drop("__remove__")


def partitioned_save(df: DataFrame, batch_size: int, max_batch: int, output: str):

    total_rows: int = df.count()
    partitions: int = max(1, min(math.ceil(total_rows / batch_size), max_batch))

    (
        df.repartition(partitions)
        .withColumn("__pid__", F.spark_partition_id())
        .write.partitionBy("__pid__")
        .parquet(output, mode="overwrite")
    )


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(description="""Inter-dataset Near-deduplicating with PySpark""")
    parser.add_argument("--input", "-i", type=str, required=True, help="GCS input directory of parquet files")
    parser.add_argument("--index", type=str, help="GCS input directory of index files")
    parser.add_argument("--index_ref", type=str, help="GCS input directory of reference index files")
    parser.add_argument("--output", type=str, required=True, help="GCS output directory of parquet files")
    parser.add_argument("--output_index", type=str, help="GCS output directory of index files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    conf = SparkConf()
    conf.set("spark.app.name", "MinHashLSH")
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()  # type: ignore
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore
    start_time = time.time()

    df: DataFrame = spark.read.option("mergeSchema", "true").parquet(args.input).cache()
    idx: DataFrame = spark.read.option("mergeSchema", "true").parquet(args.index).cache()
    index_ref: DataFrame = spark.read.option("mergeSchema", "true").parquet(args.index_ref).drop("__id__").cache()

    DATA_SIZE: int = df.count()
    MAX_WRITE_PARTITIONS: int = 256
    WRITE_ROWS: int = 1_000_000

    df_ids: Set[int] = set(df.select("__id__").distinct().collect())  # type: ignore
    idx_ids: Set[int] = set(idx.select("__id__").distinct().collect())  # type: ignore
    assert df_ids == idx_ids, f"__id__ mismatch: {df_ids - idx_ids}"

    to_remove: DataFrame = index_ref.join(idx, on="__key__", how="inner").select(["__id__"]).distinct().cache()
    index_ref.unpersist()

    df = remove_rows(df, to_remove).cache()
    idx = remove_rows(idx, to_remove).cache()
    to_remove.unpersist()

    if args.debug:
        log.debug("-" * 120)
        log.debug(f"Number of rows in B:                            {DATA_SIZE}")
        log.debug(f"Number of duplicates:                           {to_remove.count()}")
        log.debug(f"Number of remaining rows in B:                  {df.count()}")
        log.debug(f"Number of remaining rows in B (index):          {idx.count()}")
        log.debug("-" * 120)

    partitioned_save(df, WRITE_ROWS, MAX_WRITE_PARTITIONS, args.output)
    partitioned_save(idx, WRITE_ROWS, MAX_WRITE_PARTITIONS, args.output_index)

    log.debug("-" * 120)
    log.debug(f"Data Output:                            {args.output}")
    log.debug(f"Index Output:                           {args.output_index}")
    log.debug(f"Time:                                   {time.time() - start_time:.2f}s")
    log.debug("-" * 120)
