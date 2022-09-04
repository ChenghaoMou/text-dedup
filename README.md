# text-dedup

[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade)


## Features

-   Hash-based methods such as [SimHash](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf), [MinHash](https://web.archive.org/web/20150131043133/http://gatekeeper.dec.com/ftp/pub/dec/SRC/publications/broder/positano-final-wpnums.pdf) + [LSH](http://infolab.stanford.edu/~ullman/mmds.html) for near deduplication.
-   [SuffixArray](http://dl.acm.org/citation.cfm?id=320176.320218)-based method from [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) for substring exact deduplication.
-   In-memory or [Redis](https://redis.io)/[KeyDB](https://docs.keydb.dev)-cached index to handle larger than memory datasets.

## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/index.html)

## Todos
-   [ ] Memory benchmark for streaming processing
-   [ ] Speed benchmark for in-memory processing
-   [ ] Inter-dataset deduplication
-   [ ] Rewrite suffix array in Python

## Thanks

-   [seomoz/simhash-cpp](https://github.com/seomoz/simhash-cpp)
-   [datasketch](http://ekzhu.com/datasketch/index.html)
-   [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
-   Developed with OSS license from [JetBrains](https://jb.gg/OpenSourceSupport)
-   This project is heavily influenced by the deduplication work at BigScience workshop. The original code can be found at [bigscience-workshop/data-preparation](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/filtering/deduplicate).
