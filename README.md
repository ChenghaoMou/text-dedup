```
python -m text_dedup.suffixarray --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --google_repo_path "/Users/chenghao/Downloads/Projects/text-dedup/deduplicate-text-datasets"

python -m text_dedup.minhash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --ngram 1 --num_perm 128 --threshold 0.8 --seed 42

python -m text_dedup.simhash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --ngram 6  --seed 42 --bit_diff 3 --num_bucket 4

python -m text_dedup.exacthash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --dedup_name "oscar_gl_dedup" --column "text"

python -m text_dedup.bloomfilter --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --dedup_name "oscar_gl_dedup" --error_rate 1e-5 --column "text"
```
