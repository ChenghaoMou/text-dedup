<center><img src="./banner.png"/ style="background-color:white;"></center>

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![GitHub](https://img.shields.io/github/license/ChenghaoMou/text-dedup) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![DOI](https://zenodo.org/badge/347428086.svg)](https://zenodo.org/badge/latestdoi/347428086)

## Installation

```bash
git clone https://github.com/ChenghaoMou/text-dedup
cd text-dedup
uv sync
```

## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/)

## Features

This repository contains a collection of text deduplication scripts that are ready to use, or modify based on your needs:

- MinHash + MinHashLSH for near-duplicate detection
- 64 or 128 bit SimHash
- SuffixArray Substring exact deduplication
- Bloom Filter exact deduplication

All algorithms use a config-based approach with TOML files for easy customization.

## Quick Start

All deduplication scripts read from a `config.toml` file in the project root.

### 1. Configure your settings

Edit `config.toml` with your input data and algorithm settings:

<details>
<summary>MinHash Near Deduplication</summary>

```toml
[input]
input_type = "local_files"
file_type = "parquet"

[input.read_arguments]
path = "data/your_data"
split = "train"

[algorithm]
algorithm_name = "minhash"
text_column = "text"
seed = 42
batch_size = 10000
num_perm = 240
threshold = 0.7
false_positive_weight = 0.5
false_negative_weight = 0.5
hash_bits = 64
ngram_size = 5
check_false_positive = true

[output]
output_dir = "output"
clean_cache = false
save_clusters = true

[debug]
enable_profiling = false
```

</details>

<details>
<summary>SimHash Near Deduplication</summary>

```toml
[input]
input_type = "local_files"
file_type = "parquet"

[input.read_arguments]
path = "data/your_data"
split = "train"

[algorithm]
algorithm_name = "simhash"
text_column = "text"
hash_bits = 64
ngram_size = 3
bit_diff = 3

[output]
output_dir = "output"
clean_cache = false

[debug]
enable_profiling = false
```

</details>

<details>
<summary>Bloom Filter Exact Deduplication</summary>

```toml
[input]
input_type = "local_files"
file_type = "parquet"

[input.read_arguments]
path = "data/your_data"
split = "train"

[algorithm]
algorithm_name = "bloom_filter"
text_column = "text"
error_rate = 1e-5
expected_elements = 100000

[output]
output_dir = "output"
clean_cache = false

[debug]
enable_profiling = false
```

</details>

<details>
<summary>Suffix Array Substring Exact Deduplication</summary>

```toml
[input]
input_type = "local_files"
file_type = "parquet"

[input.read_arguments]
path = "data/your_data"
split = "train"

[algorithm]
algorithm_name = "suffix_array"
text_column = "text"
google_repo_path = "third_party/deduplicate-text-datasets"
merge_strategy = "longest"
length_threshold = 100
cache_dir = ".cache"

[output]
output_dir = "output"
clean_cache = false

[debug]
enable_profiling = false
```

</details>

### 2. Run the deduplication

```bash
# MinHash
python -m text_dedup.minhash

# SimHash
python -m text_dedup.simhash

# Bloom Filter
python -m text_dedup.bloom_filter

# Suffix Array
python -m text_dedup.suffix_array
```

## Benchmarks

<details>
<summary>pinecone/core-2020-05-10-deduplication</summary>

| Algorithm                       | Precision (Duplicates) | Recall (Duplicates) | Precision (Non Duplicates) | Recall (Non Duplicates) | Macro F1 score |   Accuracy | Time    |
| :------------------------------ | ---------------------: | ------------------: | -------------------------: | ----------------------: | -------------: | ---------: | :------ |
| MinHash                         |                 0.9587 |              0.9416 |                     0.9450 |                  0.9611 |     **0.9518** | **0.9277** | 11.09s  |
| SimHash                         |                 0.9038 |              0.7323 |                     0.7993 |                  0.9318 |         0.8515 |     0.8375 | 626.11s |
| Exact Title Matching [^1]       |                  0.830 |                0.50 |                      0.709 |                   0.992 |          0.757 |      0.746 | -       |
| Simhash Matching [^1]           |                  0.697 |               0.247 |                      0.598 |                   0.985 |          0.631 |      0.616 | -       |
| Document Vector Similarity [^1] |                  0.912 |               0.779 |                      0.861 |                   0.986 |          0.885 |      0.883 | -       |
| Hybrid Method [^1]              |                  0.908 |               0.828 |                      0.899 |                   0.979 |          0.904 |      0.903 | -       |
| LaBSE[^2]                       |                  0.937 |               0.923 |                      0.930 |                   0.943 |          0.933 |      0.919 | -       |
| Multilingual USE[^2]            |                  0.917 |               0.907 |                      0.918 |                   0.927 |          0.917 |      0.909 | -       |
| Multilingual E5-Base[^2]        |                  0.931 |               0.908 |                      0.919 |                   0.939 |          0.924 |      0.920 | -       |
| MinHash + LSH[^2]               |                  0.929 |               0.902 |                      0.915 |                   0.938 |          0.921 |      0.918 | -       |
| RETSim Partial-Dup[^2]          |                  0.945 |               0.941 |                      0.945 |                   0.949 |          0.945 |      0.928 | -       |
| RETSim Near-Dup[^2]             |                  0.928 |               0.937 |                      0.942 |                   0.934 |          0.935 |      0.926 | -       |

</details>
<details>
<summary>NEWS-COPY</summary>

Adjusted Rand Index (ARI) on NEWS-COPY dataset:

| Model/Algorithm          | ARI       | Time    |
| :----------------------- | :-------- | :------ |
| MinHash                  | 0.7293    | 3.01s   |
| SimHash                  | 0.6463    | 140.03s |
| n-gram [^3]              | 0.440     | -       |
| SimHash[^2]              | 0.695     | -       |
| MinHash[^3]              | 0.737     | -       |
| MinHash[^2]              | 0.783     | -       |
| Multilingual USE[^2]     | 0.730     | -       |
| Multilingual E5-Base[^2] | 0.742     | -       |
| S-BERT[^3]               | 0.700     | -       |
| RETSim Partial-Dup[^2]   | 0.831     | -       |
| RETSim Near-Dup[^2]      | 0.704     | -       |
| Re-ranking [^3]          | **0.937** | -       |
| Bi-encoder [^3]          | 0.915     | -       |

</details>

### Running Benchmarks

You can reproduce the benchmark results using the provided benchmark suite.

#### Quick Start with Just

```bash
# Run all benchmarks (both datasets, all algorithms)
just benchmark-all

# Run only CORE dataset benchmarks
just benchmark-core

# Run only NEWS-COPY dataset benchmarks
just benchmark-news

# Run specific algorithm on specific dataset
just benchmark-core-minhash
just benchmark-core-simhash
just benchmark-news-minhash
just benchmark-news-simhash
```

#### Configuration Files

Benchmark configuration files are located in `configs/`:

- `benchmark_core_minhash.toml` - MinHash on CORE dataset
- `benchmark_core_simhash.toml` - SimHash on CORE dataset
- `benchmark_news_minhash.toml` - MinHash on NEWS-COPY dataset
- `benchmark_news_simhash.toml` - SimHash on NEWS-COPY dataset

To customize benchmark parameters, edit the config files and adjust hyperparameters like `num_perm`, `threshold`, `ngram_size`, or `bit_diff`.

[^1]: [Deduplication of Scholarly Documents using Locality Sensitive Hashing and Word Embeddings](https://aclanthology.org/2020.lrec-1.113)
[^2]: [RETSim: Resilient and Efficient Text Similarity](https://arxiv.org/abs/2311.17264)
[^3]: [Noise-Robust De-Duplication at Scale](https://www.semanticscholar.org/paper/Noise-Robust-De-Duplication-at-Scale-Silcock-D'Amico-Wong/7ca41cc5fc364b713aba5b573ae4ada801fd788a)

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)

## Citations

Generally, you can cite this repository as:

```bibtex
@software{chenghao_mou_2023_8364980,
  author       = {Chenghao Mou and
                  Chris Ha and
                  Kenneth Enevoldsen and
                  Peiyuan Liu},
  title        = {ChenghaoMou/text-dedup: Reference Snapshot},
  month        = sep,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {2023.09.20},
  doi          = {10.5281/zenodo.8364980},
  url          = {https://doi.org/10.5281/zenodo.8364980}
}
```

## Acknowledgements

This repository is inspired by the following projects, and is heavily influenced by lessons learned from my own participation in [BigScience (Apache 2.0)](https://github.com/bigscience-workshop) and [BigCode (Apache 2.0)](https://github.com/bigcode-project). There is a [blog post](https://publish.obsidian.md/chenghao/posts/20230220150602) about the journey. Feedbacks are welcome!

- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)
