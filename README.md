<center><img src="./banner.png"/ style="background-color:white;"></center>

![GitHub](https://img.shields.io/github/license/ChenghaoMou/text-dedup) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![DOI](https://zenodo.org/badge/347428086.svg)](https://zenodo.org/badge/latestdoi/347428086)

## Installation

Only tested with Python 3.10 so far.

```bash
pip install text-dedup
```

or

```bash
pip install git+https://github.com/ChenghaoMou/text-dedup
```

## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/)

## Features

This repository contains a collection of text deduplication scripts that are ready to use, or modify based on your needs:

- RETSim/UniSim, an embedding-based near deduplication (WIP)
- MinHash + MinHashLSH, including a spark implementation suitable for large (TB) datasets
- 64 or 128 bit SimHash
- SuffixArray Substring
- Bloom Filter
- Exact Hash (document-level, line-level/ccnet)

I also have big plans for the future:

- [ ] Memory benchmark for streaming processing
- [ ] Inter-dataset deduplication
- [ ] Rewrite suffix array in Python
- [ ] A collections of other deduplication methods: SuperMinHash, ProbMinHash, TreeMinHash, BagMinHash, [Optimal Densification for Fast and Accurate Minwise Hashing](https://arxiv.org/abs/1703.04664), [Fast Similarity Sketching](https://arxiv.org/abs/1704.04370)

However, I do not intent to build a general purpose deduplication library, which was the goal of this repo early on. I will gradually retire the pypi package as well. The reason behind it is that each use-case can be wildly different and requires careful design and consideration. I sincerely encourage you to read the script first (they are relatively short) so you can understand what are at stake here when using it. You can use it to bootstrap your own script, or just use it as a reference.

## Acknowledgements

This repository is inspired by the following projects, and is heavily influenced by lessons learned from my own participation in [BigScience (Apache 2.0)](https://github.com/bigscience-workshop) and [BigCode (Apache 2.0)](https://github.com/bigcode-project). There is a [blog post](https://publish.obsidian.md/chenghao/posts/20230220150602) about the journey. Feedbacks are welcome!

- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)

## Quick Examples

<details>

<summary>Native PySpark</summary>

_MODIFY `text_dedup/minhash_spark.py` FOR YOUR OWN PROJECT AND DATASET FIRST!_

Assuming you have a downloaded dataset (in parquet files) under "./temp-data", you can process with file with your local compute by:

```bash
export PYSPARK_PYTHON="path to your python with scipy, xxhash, and numpy installed"
spark-submit --executor-memory 16g \
    --driver-memory 20g \
    --executor-cores 3 \
    --num-executors 2 \
    --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12 \
    --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties" \
    text_dedup/minhash_spark.py\
    --input "./temp-data" \
    --output "./temp-output" \
    --column "text" \
    --threshold 0.7
```

```
DEBUG __main__ - ------------------------------------------------------------------------------------------------------------------------
DEBUG __main__ - Using B=25, R=10
DEBUG __main__ - Loaded documents: 88803
DEBUG __main__ - args.input='./temp-data'
DEBUG __main__ - args.output='./temp-output'
DEBUG __main__ - args.threshold=0.7
DEBUG __main__ - args.ngram_size=5
DEBUG __main__ - args.min_length=5
DEBUG __main__ - args.num_perm=250
DEBUG __main__ - args.column='text'
DEBUG __main__ - id                                                              : bigint
DEBUG __main__ - text                                                            : string
DEBUG __main__ - meta                                                            : struct<warc_headers:struct<warc-record-id:string,warc-date:string,content-type:string,content-length:int,warc-type:string,warc-identified-content-language:string,warc-refers-to:string,warc-target-uri:string,warc-block-digest:string>,identification:struct<label:string,prob:float>,annotations:array<string>,line_identifications:array<struct<label:string,prob:float>>>
DEBUG __main__ - __id__                                                          : bigint
DEBUG __main__ - ------------------------------------------------------------------------------------------------------------------------
DEBUG __main__ - Initial edges: 52102
DEBUG __main__ - Edges DataFrame: 52102
DEBUG __main__ - Vertices DataFrame: 50206
DEBUG __main__ - Assignment DataFrame: 50206
DEBUG __main__ - Merging records: 88803
INFO  __main__ - Saving with 1 partitions and 44092 rows each
DEBUG __main__ - ------------------------------------------------------------------------------------------------------------------------
DEBUG __main__ - Number of rows before:    88803
DEBUG __main__ - Number of rows after:     44092
DEBUG __main__ - Percentage of rows kept:  49.65%
DEBUG __main__ - Output:                   ./temp-output
DEBUG __main__ - Time:                     68.80s
DEBUG __main__ - ------------------------------------------------------------------------------------------------------------------------

```

Or take a look at [bigcode-v2/run.sh](https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/bigcode-v2/run.sh) on how to run the job with GCP DataProc.

</details>

<details>

<summary>UniSim (WIP)</summary>

Based on Google's RETSim model([Github](https://github.com/google/unisim), [Arxiv](https://arxiv.org/abs/2311.17264)), it is an embedding based on near-deduplication method.

For a large dataset, it would require GPU(s) for fast inference.

```bash
python text_dedup/ann_unisim.py --path truthful_qa --name generation --split validation --output temp --column question
```

Output:

```
INFO     Load Dataset                    : 5.56s
INFO     Index Dataset                   : 8.13s
INFO     Clustering                      : 8.72s
INFO     Filtering                       : 0.35s
INFO     Saving                          : 0.01s
INFO     Cleaning                        : 0.00s
INFO     Total                           : 22.77s
INFO     Before                          : 817
INFO     After                           : 788
```

</details>

<details>

<summary>Suffix Array Substring Exact Deduplication</summary>

```bash
# input
python -m text_dedup.suffix_array \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output "output/suffix_array/oscar_gl_dedup" \
    --column "text" \
    --google_repo_path "/Users/chenghao/Downloads/Projects/text-dedup/deduplicate-text-datasets" \
    --use_auth_token true

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

</details>
<details>

<summary>MinHash Near Deduplication</summary>

```bash
# input
python -m text_dedup.minhash \
  --path "oscar-corpus/OSCAR-2201" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/minhash/oscar_gl_dedup" \
  --column "text" \
  --batch_size 10000 \
  --use_auth_token true

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

</details>

<details>
<summary>SimHash Near Deduplication</summary>

```bash
# input
python -m text_dedup.simhash \
  --path "oscar-corpus/OSCAR-2201" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/simhash/oscar_gl_dedup" \
  --column "text" \
  --batch_size 10000 \
  --use_auth_token true

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

</details>

<details>
<summary>Exact Hash Exact Deduplication</summary>

```bash
# input
python -m text_dedup.exact_hash \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output "output/exact_hash/oscar_gl_dedup" \
    --column "text" \
    --batch_size 1000 \
    --use_auth_token true

# output
INFO     Loading                       : 2.95s
INFO     Processing                    : 3.79s
INFO     Filtering                     : 0.10s
INFO     Saving                        : 2.89s
INFO     Total                         : 9.72s
INFO     Before                        : 88803
INFO     After                         : 47049
```

</details>

<details>
<summary>Bloom Filter Exact Deduplication</summary>

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
    --use_auth_token true    --batch_size 1000

# output
INFO     Loading                       : 2.72s
INFO     Processing                    : 4.84s
INFO     Filtering                     : 0.10s
INFO     Saving                        : 2.88s
INFO     Total                         : 10.54s
INFO     Before                        : 88803
INFO     After                         : 47045
```

</details>

## Benchmarks

> [!note]
> Spark implementation has some overhead for small datasets, so I recommend using the script only when you have a large dataset and enough compute resources.

<details>
<summary>pinecone/core-2020-05-10-deduplication</summary>

See `tests/benchmark_core.py` for reproduction.

| Algorithm                       | Precision (Duplicates) | Recall (Duplicates) | Precision (Non Duplicates) | Recall (Non Duplicates) | Macro F1 score |  Accuracy | Time     |
| :------------------------------ | ---------------------: | ------------------: | -------------------------: | ----------------------: | -------------: | --------: | :------- |
| UniSim                          |                 0.9307 |              0.8924 |                     0.9055 |                  0.9394 |         0.9181 |    0.9054 | 1305.79s |
| MinHash Spark                   |                  0.957 |              0.9445 |                     0.9471 |                   0.959 |          0.952 |    0.9202 | 691.77s  |
| MinHash                         |                 0.9594 |              0.9445 |                     0.9474 |                  0.9616 |     **0.9534** |     0.924 | 18.88s   |
| SimHash                         |                 0.9042 |               0.721 |                      0.792 |                  0.9329 |         0.8481 |    0.8321 | 644.36s  |
| Exact Title                     |                 0.8302 |              0.5521 |                     0.7098 |                  0.9065 |           0.77 |    0.7456 | -        |
| Exact Title Matching [^1]       |                  0.830 |                0.50 |                      0.709 |                   0.992 |          0.757 |     0.746 | -        |
| Simhash Matching [^1]           |                  0.697 |               0.247 |                      0.598 |                   0.985 |          0.631 |     0.616 | -        |
| Document Vector Similarity [^1] |                  0.912 |               0.779 |                      0.861 |                   0.986 |          0.885 |     0.883 | -        |
| Hybrid Method [^1]              |                  0.908 |               0.828 |                      0.899 |                   0.979 |          0.904 |     0.903 | -        |
| LaBSE[^2]                       |                  0.937 |               0.923 |                      0.930 |                   0.943 |          0.933 |     0.919 | -        |
| Multilingual USE[^2]            |                  0.917 |               0.907 |                      0.918 |                   0.927 |          0.917 |     0.909 | -        |
| Multilingual E5-Base[^2]        |                  0.931 |               0.908 |                      0.919 |                   0.939 |          0.924 |     0.920 | -        |
| MinHash + LSH[^2]               |                  0.929 |               0.902 |                      0.915 |                   0.938 |          0.921 |     0.918 | -        |
| RETSim Partial-Dup[^2]          |                  0.945 |               0.941 |                      0.945 |                   0.949 |          0.945 | **0.928** | -        |
| RETSim Near-Dup[^2]             |                  0.928 |               0.937 |                      0.942 |                   0.934 |          0.935 | **0.926** | -        |

</details>
<details>
<summary>NEWS-COPY</summary>

See `tests/benchmark_news.py` for reproduction.

Adjusted Rand Index (ARI) on NEWS-COPY dataset:

| Model/Algorithm          | ARI       |
| :----------------------- | :-------- |
| SimHash                  | 0.612     |
| MinHash (Spark)          | 0.740     |
| MinHash                  | 0.742     |
| RETSim Near-Dup + ANN\*  | _0.051_   |
| n-gram [^3]              | 0.440     |
| SimHash[^2]              | 0.695     |
| MinHash[^3]              | 0.737     |
| MinHash[^2]              | 0.783     |
| Multilingual USE[^2]     | 0.730     |
| Multilingual E5-Base[^2] | 0.742     |
| S-BERT[^3]               | 0.700     |
| RETSim Partial-Dup[^2]   | 0.831     |
| RETSim Near-Dup[^2]      | 0.704     |
| Re-ranking [^3]          | **0.937** |
| Bi-encoder [^3]          | 0.915     |

\*: I can't seem to reproduce the results from the paper.

[^1]: [Deduplication of Scholarly Documents using Locality Sensitive Hashing and Word Embeddings](https://aclanthology.org/2020.lrec-1.113)
[^2]: [RETSim: Resilient and Efficient Text Similarity](https://arxiv.org/abs/2311.17264)
[^3]: [Noise-Robust De-Duplication at Scale](https://www.semanticscholar.org/paper/Noise-Robust-De-Duplication-at-Scale-Silcock-D'Amico-Wong/7ca41cc5fc364b713aba5b573ae4ada801fd788a)

</details>

<!-- ## FAQ

### Why use scripts instead of OOD classes and functions?

Early versions of the code uses object-oriented design for hashing and indexing, which was very difficult because different methods share little to no abstraction. In order to complie something that is useful, a lot of the wrapper code was used, and that actually increased the overhead of using this library. Additionally, deduplicating is often a one-time thing in data preprocessing pipeline, there isn't really a need for inline access. -->

<!-- ### Why license change?

Because the google repo is licensed under Apache 2.0, I have to update from MIT. Util that part of code is completely re-implemented, Apache 2.0. will be the license I use. -->

## License

[Apache 2.0](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE%2D2.0.html&rut=617d395c7a807de85e5707aca1f765e5b69a1627ed84c0aefa950e54e00a3094)

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

The spark version was born from [BigCode (Apache 2.0)](https://github.com/bigcode-project) and [BigScience (Apache 2.0)](https://github.com/bigscience-workshop), and you can cite the original paper if you want:

```bibtex
@article{
kocetkov2023the,
title={The Stack: 3 {TB} of permissively licensed source code},
author={Denis Kocetkov and Raymond Li and Loubna Ben allal and Jia LI and Chenghao Mou and Yacine Jernite and Margaret Mitchell and Carlos Mu{\~n}oz Ferrandis and Sean Hughes and Thomas Wolf and Dzmitry Bahdanau and Leandro Von Werra and Harm de Vries},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=pxpbTdUEpD},
note={}
}
```
