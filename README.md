
# text-dedup

A collection of data deduplication scripts.



![GitHub](https://img.shields.io/github/license/ChenghaoMou/text-dedup) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ChenghaoMou/text-dedup&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage)
## Features

- Ready to use and modify single script for each method:
    - MinHash + MinHashLSH
    - SimHash
    - SuffixArray Substring
    - Bloom Filter
    - Exact Hash

## Acknowledgements

 - [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
 - [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
 - [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
 - [BigScience](https://github.com/bigscience-workshop) (Apache 2.0)
 - [BigCode](https://github.com/bigcode-project) (Apache 2.0)


## Quick Examples

In this section, we are going to deduplicate one dataset: `gl` subset of `oscar-corpus/OSCAR-2201`.

### Suffix Array Substring Exact Deduplication
```bash
# input
python -m text_dedup.suffix_array \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output_dir "output/suffix_array" \
    --index_name "lsh.pkl" \
    --graph_name "graph.networkit" \
    --dedup_name "oscar_gl_dedup" \
    --column "text" \
    --google_repo_path "/Users/chenghao/Downloads/Projects/text-dedup/deduplicate-text-datasets"

# output
INFO     All                           : 131.93s
INFO     Loading                       : 4.36s
INFO     Preprocessing                 : 4.81s
INFO     Suffix Array                  : 101.79s
INFO     Collect                       : 5.17s
INFO     Restore                       : 0.27s
INFO     Deduplicate                   : 13.00s
INFO     Saving                        : 2.52s
INFO     Before                        : 180332342 bytes (88803)
INFO     After                         : 97646271 bytes (40404)
INFO     Output                        : output/suffix_array/oscar_gl_dedup
```

### MinHash Near Deduplication
```bash
# input
python -m text_dedup.minhash \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output_dir "output/minhash" \
    --index_name "lsh.pkl" \
    --graph_name "graph.networkit" \
    --dedup_name "oscar_gl_dedup" \
    --column "text" \
    --ngram 1 \
    --num_perm 128 \
    --threshold 0.8 \
    --seed 42

# output
INFO     All                           : 52.73s
INFO     Loading                       : 5.32s
INFO     Minhash                       : 12.82s
INFO     Index                         : 8.54s
INFO     Save Index                    : 3.86s
INFO     Query                         : 4.49s
INFO     Clustering                    : 17.47s
INFO     Deduplicate                   : 0.05s
INFO     Save                          : 0.04s
INFO     Before                        : 88803
INFO     After                         : 43971
INFO     Index                         : output/minhash/lsh.pkl
INFO     Graph                         : output/minhash/graph.networkit
INFO     Output                        : output/minhash/oscar_gl_dedup
```

### SimHash Near Deduplication
```bash
# input
python -m text_dedup.simhash \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output_dir "output/simhash" \
    --index_name "index.pkl" \
    --graph_name "graph.networkit" \
    --dedup_name "oscar_gl_dedup" \
    --column "text" \
    --ngram 6 \
    --bit_diff 3 \
    --num_bucket 4

# output
INFO     All                           : 39.88s
INFO     Loading                       : 4.45s
INFO     Simhash                       : 1.91s
INFO     Index                         : 5.23s
INFO     Save Index                    : 1.44s
INFO     Query                         : 6.57s
INFO     Clustering                    : 16.42s
INFO     Deduplicate                   : 0.72s
INFO     Save                          : 3.11s
INFO     Before                        : 88803
INFO     After                         : 46659
INFO     Index                         : output/simhash/index.pkl
INFO     Graph                         : output/simhash/graph.networkit
INFO     Output                        : output/simhash/oscar_gl_dedup
```

### Exact Hash Exact Deduplication
```bash
# input
python -m text_dedup.exact_hash \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output_dir "output/exact_hash" \
    --dedup_name "oscar_gl_dedup" \
    --column "text"

# output
INFO     All                           : 5.34s
INFO     Loading                       : 4.48s
INFO     Processing                    : 0.73s
INFO     Filtering                     : 0.07s
INFO     Saving                        : 0.05s
INFO     Before                        : 88803
INFO     After                         : 47049
```

### Bloom Filter Exact Deduplication
```bash
# input
python -m text_dedup.bloom_filter \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output_dir "output/bloom_filter" \
    --dedup_name "oscar_gl_dedup" \
    --error_rate 1e-5 \
    --column "text"

# output
INFO     All                           : 10.69s
INFO     Loading                       : 4.44s
INFO     Processing                    : 6.13s
INFO     Filtering                     : 0.07s
INFO     Saving                        : 0.05s
INFO     Before                        : 88803
INFO     After                         : 47045
```


## Documentation

- [ ] TODO
## Roadmap

-   [ ] Memory benchmark for streaming processing
-   [ ] Speed benchmark for in-memory processing
-   [ ] Inter-dataset deduplication
-   [ ] Rewrite suffix array in Python
-   [ ] A collections of deduplication methods used in papers/datasets/projects
-   [ ] SuperMinHash, ProbMinHash, TreeMinHash, BagMinHash, [Optimal Densification for Fast and Accurate Minwise Hashing](https://arxiv.org/abs/1703.04664), [Fast Similarity Sketching](https://arxiv.org/abs/1704.04370)
## FAQ

### Why use scripts instead of OOD classes and functions?

Early versions of the code uses object-oriented design for hashing and indexing, which was very difficult because different methods share little to no abstraction. In order to complie something that is useful, a lot of the wrapper code was used, and that actually increased the overhead of using this library. Additionally, deduplicating is often a one-time thing in data preprocessing pipeline, there isn't really a need for inline access.


### Why license change?

Because the google repo is licensed under Apache 2.0, I have to update from MIT. Util that part of code is completely re-implemented, Apache 2.0. will be the license I use.


## License

[Apache 2.0](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE%2D2.0.html&rut=617d395c7a807de85e5707aca1f765e5b69a1627ed84c0aefa950e54e00a3094)
