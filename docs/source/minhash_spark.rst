MinHash + LSH (Spark)
=====================

This extends the MinHash + LSH implementation to work with Spark, specifically, GCP dataproc, see :mod:`text_dedup.minhash` (:doc:`/minhash`) for more details.

Quick Start
-----------

.. code-block:: bash

   export CLUSTER_NAME=chenghao-temp
   export PROJECT_ID=xx

   gcloud dataproc clusters create $CLUSTER_NAME \
      --enable-component-gateway \
      --region us-central1 \
      --zone us-central1-a \
      --master-machine-type c2d-standard-16 \
      --master-boot-disk-size 500 \
      --num-workers 10 \
      --worker-machine-type c2d-standard-16 \
      --worker-boot-disk-size 500 \
      --image-version 2.0-debian10 \
      --project $PROJECT_ID

   gcloud dataproc jobs submit pyspark --cluster ${CLUSTER_NAME} \
      --region us-central1 \
      --jars gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
      --driver-log-levels root=WARN \
      --properties="spark.executor.memory"="50g","spark.driver.memory"="8g","spark.executor.cores"="14" \
      minhash_spark.py -- \
      --table "huggingface-science-codeparrot.the_stack_java.java" \
      --output "gs://chenghao-data/dataproc_output/deduplicated"

.. automodule:: text_dedup.minhash_spark
   :members:
   :undoc-members:
   :noindex:

