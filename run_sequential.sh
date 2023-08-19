DATA_NAME=$1
NUM_WORKERS=100
BATCH_SIZE=10000

name=${DATA_NAME##*/}

python -m text_dedup.fix_text \
  --path "${DATA_NAME}" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${name}/fixed" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --min_words 5 \
  --not_shuffle \
  --use_auth_token \
  "${@:2}"

python -m text_dedup.exact_hash \
  --path "/workspace/datasets/${name}/fixed" \
  --local \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${name}/dedup_exact" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --debug

#rm -rf /workspace/datasets/${DATA_NAME}/fixed

python -m text_dedup.minhash \
  --path "/workspace/datasets/${name}/dedup_exact" \
  --local \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/${name}/dedup_minhash" \
  --column "text" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --debug
