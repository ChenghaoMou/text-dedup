<<<<<<< HEAD
# text-dedup

> **Warning**
> Breaking changes will happen very frequently before 1.0. This is also a learning process for me. Please proceed with caution. If you want to use a one-time version, you can check out this https://github.com/bigcode-project/bigcode-analysis/blob/1fe56970240f7547e4bc92f4bc23e0470bdbb9aa/data_analysis/near-deduplication/minhash_deduplication_alt.py.

[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade)


## Features

-   Hash-based methods such as [SimHash](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf), [MinHash](https://web.archive.org/web/20150131043133/http://gatekeeper.dec.com/ftp/pub/dec/SRC/publications/broder/positano-final-wpnums.pdf) + [LSH](http://infolab.stanford.edu/~ullman/mmds.html) for near deduplication.
-   [SuffixArray](http://dl.acm.org/citation.cfm?id=320176.320218)-based method from [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) for substring exact deduplication.
-   In-memory or [Redis](https://redis.io)/[KeyDB](https://docs.keydb.dev)-cached index to handle larger than memory datasets.

## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/index.html)

## Todos

-   [ ] Minimize extra code footprint
-   [ ] Memory benchmark for streaming processing
-   [ ] Speed benchmark for in-memory processing
-   [ ] Inter-dataset deduplication
-   [ ] Rewrite suffix array in Python
-   [ ] Rewrite simhash with bitarray
-   [ ] Rewrite minhash without datasketch
-   [ ] A collections of deduplication methods used in papers/datasets/projects
-   [ ] SuperMinHash, ProbMinHash, TreeMinHash, BagMinHash, [Optimal Densification for Fast and Accurate Minwise Hashing](https://arxiv.org/abs/1703.04664), [Fast Similarity Sketching](https://arxiv.org/abs/1704.04370)

## Thanks

-   [seomoz/simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
-   [datasketch](http://ekzhu.com/datasketch/index.html) (MIT)
-   [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) (Apache-2.0)
-   Developed with OSS license from [JetBrains](https://jb.gg/OpenSourceSupport)
-   This project is heavily influenced by the deduplication work at BigScience workshop. The original code can be found at [bigscience-workshop/data-preparation](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/filtering/deduplicate) (Apache-2.0)

## License

License was changed from MIT to Apache 2.0 on Oct 2, 2022 to be consistent with the libraries used in this project.

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)
=======
```
python -m text_dedup.suffixarray --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --google_repo_path "/Users/chenghao/Downloads/Projects/text-dedup/deduplicate-text-datasets"

python -m text_dedup.minhash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --ngram 1 --num_perm 128 --threshold 0.8 --seed 42

python -m text_dedup.simhash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --index_name "lsh.pkl" --graph_name "graph.networkit" --dedup_name "oscar_gl_dedup" --column "text" --ngram 6  --seed 42 --bit_diff 3 --num_bucket 4

python -m text_dedup.exacthash --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --dedup_name "oscar_gl_dedup" --column "text"

python -m text_dedup.bloomfilter --path "oscar-corpus/OSCAR-2201" --name "gl" --split "train" --cache_dir "./cache" --output_dir "output" --dedup_name "oscar_gl_dedup" --error_rate 1e-5 --column "text"
```
>>>>>>> e1e34dc (update argparse)
