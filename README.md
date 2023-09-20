<center><img src="./banner.png"/ style="background-color:white;"></center>

![GitHub](https://img.shields.io/github/license/ChenghaoMou/text-dedup) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![DOI](https://zenodo.org/badge/347428086.svg)](https://zenodo.org/badge/latestdoi/347428086)


## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/)

## Features

This repository contains a collection of text deduplication scripts that are ready to use, or modify based on your needs:

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

### PySpark with DataProc

Not a lot of people have access to enough compute resources or the need to deduplicate TB-scale datasets, but if you do, this is a good example of how to use it with GCP DataProc.

*MODIFY `text_dedup/minhash_spark.py` FOR YOUR OWN PROJECT AND DATASET FIRST!*

```bash
export CLUSTER_NAME=chenghao-temp
export PROJECT_ID=xx
export REGION=us-central1
export ZONE=us-central1-a
export INPUT_GCS_PATH="gs://chenghao-temp-exp/data/ada"
export OUTPUT_GCS_PATH="gs://chenghao-temp-exp/output/ada"

gcloud dataproc clusters create $CLUSTER_NAME \
    --enable-component-gateway \
    --region $REGION \
    --zone $ZONE \
    --master-machine-type c2d-standard-16 \
    --master-boot-disk-size 500 \
    --num-workers 10 \
    --worker-machine-type c2d-standard-16 \
    --worker-boot-disk-size 500 \
    --image-version 2.0-debian10 \
    --project $PROJECT_ID

gcloud dataproc jobs submit pyspark --cluster ${CLUSTER_NAME}\
    --region $REGION \
    --jars gs://spark-lib/bigquery/spark-3.3-bigquery-0.32.2.jar \
    --driver-log-levels root=FATAL,__main__=DEBUG \
    --properties="spark.executor.memory"="50g","spark.driver.memory"="8g","spark.executor.cores"="14" \
    minhash_spark.py -- --input $INPUT_GCS_PATH --output $OUTPUT_GCS_PATH
```

For reference, the script finished deduplicating 42 million rows in less than 40 minutes with above settings (160 cores, 640GB memory in total), while the python version would take around 10 hours with a 80-core machine with 1.8TB memory.

In the following part, we are going to deduplicate one dataset: `gl` subset of `oscar-corpus/OSCAR-2201`.

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

| Method                                                                          | Precision        | Recall           | F1               | Time |
| ------------------------------------------------------------------------------- | ---------------- | ---------------- | ---------------- | ---- |
| MinHash                                                                         | **0.9464** | **0.9446** | **0.9455** | 24s  |
| SimHash\*                                                                       | 0.9011           | 0.6959           | 0.7853           | 210s |
| SimHash[(Gyawali et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.113)     | 0.697            | 0.247            | 0.3647           | -    |
| Exact Title (my implementation)                                                 | 0.8302           | 0.5521           | 0.6632           | -    |
| Exact Title[(Gyawali et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.113) | 0.830            | 0.50             | 0.624            | -    |

\*Best SimHash result from `benchmarks/hyperparameter.ipynb`.

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
