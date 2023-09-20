#!/bin/bash
# -*- coding: utf-8 -*-
# @Date    : 2023-09-02 10:30:06
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

THRESHOLD=0.7
REPO_COLUMN="repo_url"

/Users/chenghao/Downloads/spark-3.5.0-bin-hadoop3/bin/spark-submit \
    --executor-memory 20g \
    --driver-memory 10g \
    --executor-cores 2 \
    --num-executors 2 \
    --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 \
    --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
    --conf "spark.python.profile=true" \
    intra_dedup.py \
    --input "/Users/chenghao/Downloads/temp/input" \
    --output "/Users/chenghao/Downloads/temp/output" \
    --output_index "/Users/chenghao/Downloads/temp/output_index" \
    --threshold $THRESHOLD \
    --repo_column $REPO_COLUMN \
    --rank
