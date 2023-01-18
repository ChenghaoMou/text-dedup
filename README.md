# text-dedup

A collection of data deduplication scripts.

![GitHub](https://img.shields.io/github/license/ChenghaoMou/text-dedup) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage)

Intended for deduplicating moderate datasets (<100 M docs) with one multi-core machine. Designed to be modified for specific use cases.
Please reach out if you are interested in collaboration for large scale deduplication in distributed environment.

## Features

- Ready to use or modify scripts for each deduplication method:
  - MinHash + MinHashLSH
  - SimHash (64, 128)
  - SuffixArray Substring
  - Bloom Filter
  - Exact Hash

## Acknowledgements

- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [BigScience](https://github.com/bigscience-workshop) (Apache 2.0)
- [BigCode](https://github.com/bigcode-project) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)

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
    --output "output/suffix_array/oscar_gl_dedup" \
    --column "text" \
    --google_repo_path "/Users/chenghao/Downloads/Projects/text-dedup/deduplicate-text-datasets"

# output
INFO     Loading                       : 2.75 seconds
INFO     Preprocessing                 : 4.78 seconds
INFO     SuffixArray                   : 98.29 seconds
INFO     SelfSimilar                   : 4.24 seconds
INFO     Restore                       : 0.25 seconds
INFO     Deduplicate                   : 6.23 seconds
INFO     Saving                        : 8.91 seconds
INFO     Total                         : 125.45 seconds
INFO     Before                        : 180332342 bytes (88803)
INFO     After                         : 97646271 bytes (40404)
```

### MinHash Near Deduplication

```bash
# input
python -m text_dedup.minhash \
  --path "oscar-corpus/OSCAR-2201" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/minhash/oscar_gl_dedup" \
  --column "text" \
  --batch_size 10000

# output
INFO     Loading                         : 2.62 seconds
INFO     MinHashing                      : 0.08 seconds
INFO     Clustering                      : 2.20 seconds
INFO     Filtering                       : 0.53 seconds
INFO     Saving                          : 9.86 seconds
INFO     Total                           : 15.29 seconds
INFO     Data Number (before)            : 88803
INFO     Data Number (after)             : 44124 (49.69%)
INFO     Duplicate Number                : 44679 (50.31%)
INFO     🤗 Happy Deduplicating 🤗
```

### SimHash Near Deduplication

```bash
# input
python -m text_dedup.simhash \
  --path "oscar-corpus/OSCAR-2201" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/simhash/oscar_gl_dedup" \
  --column "text" \
  --batch_size 10000

# output
INFO     Loading                         : 2.60 seconds
INFO     SimHashing                      : 0.04 seconds
INFO     Indexing                        : 28.88 seconds
INFO     Filtering                       : 0.88 seconds
INFO     Saving                          : 10.41 seconds
INFO     Total                           : 42.80 seconds
INFO     Data Number (before)            : 88803
INFO     Data Number (after)             : 46163 (51.98%)
INFO     Duplicate Number                : 42640 (48.02%)
INFO     🤗 Happy Deduplicating 🤗
```

### Exact Hash Exact Deduplication

```bash
# input
python -m text_dedup.exact_hash \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output "output/exact_hash/oscar_gl_dedup" \
    --column "text" \
    --batch_size 1000

# output
INFO     Loading                       : 2.95s
INFO     Processing                    : 3.79s
INFO     Filtering                     : 0.10s
INFO     Saving                        : 2.89s
INFO     Total                         : 9.72s
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
    --output "output/bloom_filter/oscar_gl_dedup" \
    --error_rate 1e-5 \
    --column "text" \
    --batch_size 1000

# output
INFO     Loading                       : 2.72s
INFO     Processing                    : 4.84s
INFO     Filtering                     : 0.10s
INFO     Saving                        : 2.88s
INFO     Total                         : 10.54s
INFO     Before                        : 88803
INFO     After                         : 47045
```

## Benchmarks

A benchmark of different methods here can be found in `benchmarks/wiki40.ipynb`. A notebook in evaluating MinHash on `pinecone/core-2020-05-10-deduplication` can be found in `benchmarks/pinecone.ipynb`.

For quick reference, here are the results:

| Method                                                                              | Precision  | Recall     | F1         | Time |
| ----------------------------------------------------------------------------------- | ---------- | ---------- | ---------- | ---- |
| MinHash                                                                             | **0.9464** | **0.9446** | **0.9455** | 24s  |
| SimHash\*                                                                           | 0.9011     | 0.6959     | 0.7853     | 210s |
| SimHash [(Gyawali et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.113)     | 0.697      | 0.247      | 0.3647     | -    |
| Exact Title (my implementation)                                                     | 0.8302     | 0.5521     | 0.6632     | -    |
| Exact Title [(Gyawali et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.113) | 0.830      | 0.50       | 0.624      | -    |

\*Best SimHash result from `benchmarks/hyperparameter.ipynb`.

## Documentation

- [ ] TODO

## Roadmap

- [ ] Memory benchmark for streaming processing
- [ ] Inter-dataset deduplication
- [ ] Rewrite suffix array in Python
- [ ] A collections of deduplication methods used in papers/datasets/projects: SuperMinHash, ProbMinHash, TreeMinHash, BagMinHash, [Optimal Densification for Fast and Accurate Minwise Hashing](https://arxiv.org/abs/1703.04664), [Fast Similarity Sketching](https://arxiv.org/abs/1704.04370)

## FAQ

### Why use scripts instead of OOD classes and functions?

Early versions of the code uses object-oriented design for hashing and indexing, which was very difficult because different methods share little to no abstraction. In order to complie something that is useful, a lot of the wrapper code was used, and that actually increased the overhead of using this library. Additionally, deduplicating is often a one-time thing in data preprocessing pipeline, there isn't really a need for inline access.

### Why license change?

Because the google repo is licensed under Apache 2.0, I have to update from MIT. Util that part of code is completely re-implemented, Apache 2.0. will be the license I use.

## License

[Apache 2.0](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE%2D2.0.html&rut=617d395c7a807de85e5707aca1f765e5b69a1627ed84c0aefa950e54e00a3094)
