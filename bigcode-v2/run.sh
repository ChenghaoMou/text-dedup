#!/bin/bash
# -*- coding: utf-8 -*-
# @Date    : 2023-09-02 10:30:06
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

CLUSTER_NAME="chenghao-temp"
PROJECT_ID="huggingface-science-codeparrot"
REGION="us-central1"
CONTAINER=""
DIRECTORY="temp"
CHECKPOINT_DIR="hdfs:///tmp/checkpoints"
NUM_WORKERS=24
MASTER_MACHINE_TYPE="c2-standard-16"
MASTER_BOOT_DISK_SIZE=1024
WORKER_MACHINE_TYPE="c2-standard-60"
WORKER_BOOT_DISK_SIZE=2048
IMAGE_VERSION="2.0-debian10"
SPARK_JARS="gs://spark-lib/bigquery/spark-3.2-bigquery-0.32.2.jar"
THRESHOLD=0.7
REPO_COLUMN="repo_url"

DEDUPED_DIRECTORY="${DIRECTORY}_deduped"
# DEDUPED_INDEX_DIRECTORY="${DEDUPED_DIRECTORY}_index"
DIRS=("f_star")
# DIRS=$(cat dirs.list)

# Create cluster if it doesn't exist
if ! gcloud dataproc clusters list --region $REGION | grep -q $CLUSTER_NAME; then
    gcloud dataproc clusters create $CLUSTER_NAME \
        --enable-component-gateway \
        --region $REGION \
        --zone "" \
        --master-machine-type $MASTER_MACHINE_TYPE \
        --master-boot-disk-size $MASTER_BOOT_DISK_SIZE \
        --num-workers $NUM_WORKERS \
        --worker-machine-type $WORKER_MACHINE_TYPE \
        --worker-boot-disk-size $WORKER_BOOT_DISK_SIZE \
        --image-version $IMAGE_VERSION \
        --project $PROJECT_ID \
        --properties=^#^dataproc:conda.packages='scipy==1.10.1'#dataproc:pip.packages='xxhash==3.3.0'
fi

# Start cluster if it's not running
if ! gcloud dataproc clusters list --region $REGION | grep -q RUNNING | grep -q $CLUSTER_NAME; then
    gcloud dataproc clusters start $CLUSTER_NAME --region $REGION
fi

# Progress bar
TOTAL=$(echo "${DIRS}" | wc -w)
LENGTH=20
i=0

echo "Total number of directories: $TOTAL"
for DIR in $DIRS; do
    # Progress bar
    echo -n "[ "
    curr_pos=$((i * LENGTH / TOTAL))
    for ((k = 0; k <= curr_pos; k++)); do echo -n "==="; done
    for ((j = k + 1; j <= LENGTH; j++)); do echo -n "   "; done
    v=$(((i + 1) * 100 / TOTAL))
    echo -n " ] "
    echo "$v %" $'\r'
    ((i++))

    DIR=${DIR%/}
    INPUT_GCS_PATH="${CONTAINER}/${DIRECTORY}/${DIR}"
    LAN=$(echo "$DIR" | rev | cut -d'/' -f1 | rev)
    OUTPUT_GCS_PATH="${CONTAINER}/${DEDUPED_DIRECTORY}/${LAN}"
    # OUTPUT_INDEX_GCS_PATH="${CONTAINER}/${DEDUPED_INDEX_DIRECTORY}/${LAN}"
    # OUTPUT_STATUS_GCS_PATH="${OUTPUT_GCS_PATH}/_SUCCESS"
    # result=$(gsutil stat "${OUTPUT_STATUS_GCS_PATH}" 2>&1 | grep -c "No URLs matched")
    # if [[ $result != 1 ]]; then
    #     echo "Skipping ${LAN}"
    #     continue
    # fi
    echo "Processing ${LAN}"

    gcloud dataproc jobs submit pyspark --cluster ${CLUSTER_NAME} \
        --region $REGION \
        --jars $SPARK_JARS \
        --driver-log-levels root=FATAL,__main__=DEBUG \
        --properties="spark.executor.memory=210g,spark.driver.memory=16g,spark.executor.cores=59,spark.jars.packages=graphframes:graphframes:0.8.2-spark3.2-s_2.12" \
        intra_dedup.py -- \
        --input "$INPUT_GCS_PATH" \
        --output "$OUTPUT_GCS_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --threshold $THRESHOLD \
        --repo_column $REPO_COLUMN \
        --rank

done

gcloud dataproc clusters stop $CLUSTER_NAME --region $REGION
