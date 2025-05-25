MinHash + LSH (Spark)
=====================

This extends the MinHash + LSH implementation to work with Spark, specifically, GCP dataproc, see :mod:`text_dedup.minhash` (:doc:`/minhash`) for more details. I try my best to maintain the parity between the two versions.

Quick Start
-----------

.. code-block:: bash

   export CLUSTER_NAME=chenghao-temp
   export PYSPARK_PYTHON="path to your python with scipy, xxhash, and numpy installed"
   export PROJECT_ID=xx
   spark-submit --executor-memory 16g \
   export REGION=us-central1
      --driver-memory 20g \
   export ZONE=us-central1-a
      --executor-cores 3 \
   export INPUT_GCS_PATH="gs://chenghao-temp-exp/data/ada"
      --num-executors 2 \
   export OUTPUT_GCS_PATH="gs://chenghao-temp-exp/output/ada"
      --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 \

      --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
   gcloud dataproc clusters create $CLUSTER_NAME \
      --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
      --enable-component-gateway \
      text_dedup/minhash_spark.py\
      --region $REGION \
      --input "./temp-data" \
      --zone $ZONE \
      --output "./temp-output" \
      --master-machine-type c2d-standard-16 \
      --column "text" \
      --master-boot-disk-size 500 \
      --threshold 0.7 \
      --num-workers 10 \
      --debug
      --worker-machine-type c2d-standard-16 \
      --worker-boot-disk-size 500 \
      --image-version 2.0-debian10 \
      --project $PROJECT_ID

   gcloud dataproc jobs submit pyspark --cluster ${CLUSTER_NAME}\
      --region $REGION \
      --jars gs://spark-lib/bigquery/spark-3.3-bigquery-0.32.2.jar \
      --driver-log-levels root=FATAL,__main__=DEBUG \
      --properties="spark.executor.memory"="50g","spark.driver.memory"="8g","spark.executor.cores"="14" \
      minhash_spark.py -- --input $INPUT_GCS_PATH --output $OUTPUT_GCS_PATH

For reference, the script finished deduplicating 42 million rows in less than 40 minutes with above settings (160 cores, 640GB memory in total), while the python version would take around 10 hours with a 80-core machine with 1.8TB memory.

For more details on BigCode scripts, you can check out the scripts under reference/bigcode-v2.

You can also use it with native spark-submit:

.. code-block:: bash

   export PYSPARK_PYTHON="path to your python with scipy, xxhash, and numpy installed"
   spark-submit --executor-memory 16g \
      --driver-memory 20g \
      --executor-cores 3 \
      --num-executors 2 \
      --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 \
      --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
      --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
      text_dedup/minhash_spark.py\
      --input "./temp-data" \
      --output "./temp-output" \
      --column "text" \
      --threshold 0.7

.. automodule:: text_dedup.minhash_spark
   :members:
   :undoc-members:
   :noindex:
