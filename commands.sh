python -m text_dedup.minhash \
  --path "eduagarcia/iudicium_textum" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/iudicium_textum/dedup" \
  --column "text" \
  --batch_size 10000 \
  --num_workers 180 \
  --debug

python -m text_dedup.minhash \
  --path "eduagarcia/tesemo_v2" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/tesemo_v2/dedup" \
  --column "text" \
  --batch_size 10000 \
  --num_workers 180 \
  --debug

python -m text_dedup.minhash \
  --path "eduagarcia/pt_legal_pile" \
  --name "pt_all" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets/pt_legal_pile/dedup" \
  --column "text" \
  --batch_size 10000 \
  --num_workers 180

python -m text_dedup.minhash \
  --path "eduagarcia/datalawyer-v1" \
  --split "train" \
  --cache_dir "/workspace/datasets/hf_cache" \
  --output "/workspace/datasets//datalawyer-v1/dedup" \
  --column "text" \
  --batch_size 10000 \
  --num_workers 180 \
  --use_auth_token