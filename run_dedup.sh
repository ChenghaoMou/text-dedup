set -e

DATA_PATH=$1
DATA_NAME=$2
NUM_WORKERS=100
BATCH_SIZE=50000

#DATA_NAME=${DATA_PATH##*/}

python -m text_dedup.exact_hash_norm \
  --path "${DATA_PATH}" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${DATA_NAME}/exact_norm" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --use_auth_token \
  --no-filter \
  --hash_func "xxh3" \
  --debug \
  "${@:3}"

# rm -rf /workspace/datasets/${DATA_NAME}/fixed

python -m text_dedup.minhash \
  --path "/workspace/datasets/${DATA_NAME}/exact_norm" \
  --local \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${DATA_NAME}/dedup" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --filter \
  --save_both \
  --hash_func "xxh3" \
  --hash_bits 32 \
  --debug

rm -rf /workspace/datasets/${DATA_NAME}/exact_norm