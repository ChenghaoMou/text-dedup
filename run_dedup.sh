DATA_NAME=$1
NUM_WORKERS=100
BATCH_SIZE=10000

python -m text_dedup.exact_hash \
  --path "eduagarcia/${DATA_NAME}" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${DATA_NAME}/dedup_exact" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --use_auth_token \
  --debug \
  "${@:2}"

# rm -rf /workspace/datasets/${DATA_NAME}/fixed

python -m text_dedup.minhash \
  --path "/workspace/datasets/${DATA_NAME}/dedup_exact" \
  --local \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${DATA_NAME}/dedup_minhash" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --debug
