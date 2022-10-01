#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/30/22
# description : Minhash and LSH in spark
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

from datasets import load_dataset
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.feature import NGram
from pyspark.ml.feature import RegexTokenizer
from pyspark.shell import sc
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from text_dedup.base import Deduplicator

spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "32g") \
    .appName('text-dedup') \
    .getOrCreate()


@dataclass
class PySparkMinHashLSHDeduplicator(Deduplicator):

    jaccard_similarity: float = 0.85
    num_perm: int = 8
    ngram_size: int = 2
    featurizer: Pipeline = field(default=None, init=False)
    model: PipelineModel = field(default=None, init=False)
    df: DataFrame = field(default=None, init=False)

    def __post_init__(self):
        self.featurizer = Pipeline(
            stages=[
                RegexTokenizer(
                    pattern="\\W", inputCol="content", outputCol="tokens", minTokenLength=1
                ),
                NGram(n=self.ngram_size, inputCol="tokens", outputCol="ngrams"),
                HashingTF(inputCol="ngrams", outputCol="vectors"),
                MinHashLSH(inputCol="vectors", outputCol="lsh", numHashTables=self.num_perm)
            ]
        )

    def fit(self, data: Sequence[str]):
        df = self.preprocess(data)
        self.model = self.featurizer.fit(df)
        self.df = self.model.transform(df)

    def predict(self, data: Sequence[str]):
        df = self.model.transform(self.preprocess(data))
        res = self.model.stages[-1].approxSimilarityJoin(self.df, df, 1 - self.jaccard_similarity, distCol="JaccardDistance") \
            .select(col("datasetA.id").alias("idA"),
                    col("datasetB.id").alias("idB"),
                    col("JaccardDistance"))
        return res

    def fit_predict(self, data: Sequence[str]):

        self.fit(data)
        return self.predict(data)

    @staticmethod
    def preprocess(data: Sequence[str]):
        N = os.cpu_count()
        records = [
            (i, text) for i, text in enumerate(data)
        ]
        df = spark.createDataFrame(records, ["id", "content"]).repartition(N)
        return df


if __name__ == '__main__':

    import tracemalloc

    import typer
    try:
        from humanize import naturalsize as humanize_size
    except ImportError:
        def humanize_size(x):
            return f"{x} B"

    logger = logging.getLogger("text_dedup")

    def run(
            dataset: str,
            config: str,
            split: str,
            jaccard_threshold: float = 0.85,
            verbose: bool = False
    ):
        start_time = time.time()
        logging.basicConfig(level=logging.INFO)
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            sc.setLogLevel("DEBUG")
        ds = load_dataset(dataset, config, split=split, use_auth_token=True)
        deduplicator = PySparkMinHashLSHDeduplicator(jaccard_similarity=jaccard_threshold)
        res = deduplicator.fit_predict(ds["content"])

        ids = set()

        for row in res.collect():
            if row.idA == row.idB:
                continue
            ids.add(row.idA)
            ids.add(row.idB)

        logger.info(f"Total time: {time.time() - start_time}")
        logger.info(f"Total duplicates: {len(ids)}")
        pass

    tracemalloc.start()

    typer.run(run)

    _, usage = tracemalloc.get_traced_memory()

    logger.info(f"Using {humanize_size(usage)} memory")

    tracemalloc.stop()
