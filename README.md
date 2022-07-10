# text-dedup

[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade)


## Features

-   Hash-based methods such as [SimHash](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf), [MinHash](https://web.archive.org/web/20150131043133/http://gatekeeper.dec.com/ftp/pub/dec/SRC/publications/broder/positano-final-wpnums.pdf) + [LSH](http://infolab.stanford.edu/~ullman/mmds.html) for near deduplication.
-   [SuffixArray](http://dl.acm.org/citation.cfm?id=320176.320218)-based method from [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) for substring exact deduplication.
-   In-memory or [Redis](https://redis.io)/[KeyDB](https://docs.keydb.dev)-cached index to handle larger than memory datasets.

## Documentation

[Github Pages](https://chenghaomou.github.io/text-dedup/text_dedup.html)

## CLI Usage

`cli.py` is a wrapper tool that identifies duplicates for a given Huggingface's dataset. Currently, only hash-based methods will try to identify all duplicates within the dataset and the suffix array method will only find the duplicate substrings within dataset splits.

By default, the tool uses redis as a cache layer for the hashes. See `configs/method/minhash.yaml` or `configs/method/simhash.yaml` for details. Or you can overwrite the `storage_config` to `null` to use in-memory index. Deduplicating small datasets that fit in your machine's memory should be fine with in-memory index.

```text
python cli.py method=suffix  method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
python cli.py method=simhash method.tokenization.ngram_size=12 method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
python cli.py method=minhash method.tokenization.ngram_size=12 method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
```

-   Configurations are parsed with [hydra](https://hydra.cc).

## Programmatic Usage

### Hash-based Near Deduplication

```python
from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.utils.nn import lsh_clustering
from text_dedup.utils.group import get_group_indices

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog",
    "This is a test",
    "This is a test",
]

embedder = MinHashEmbedder()
embeddings = embedder.embed(corpus)

clusters = lsh_clustering(embeddings)
groups = get_group_indices(clusters)
print(groups)
# [0, 0, 2, 2]
```

```python
from text_dedup.embedders.simhash import SimHashEmbedder
from text_dedup.utils.nn import simhash_clustering
from text_dedup.utils.group import get_group_indices

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog",
    "This is a test",
    "This is a test",
]

embedder = SimHashEmbedder()
embeddings = embedder.embed(corpus)

clusters = simhash_clustering(embeddings)
groups = get_group_indices(clusters)
print(groups)
# [0, 0, 2, 2]
```

### Suffix Array Substring Exact Deduplication

```python
from text_dedup.embedders.suffix import SuffixArrayEmbedder

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog",
    "This is a test",
    "This is a test",
    "This is a random test",
    "The quick brown fox and a random test"
]


embedder = SuffixArrayEmbedder(k=10)
slices = embedder.embed(corpus, merge=True, merge_strategy='longest')
# or using the original rust code
# slices = embedder.embed_bash(corpus)

for sentence, intervals in zip(corpus, slices):
    print(sentence)
    print([sentence[slice] for slice in intervals])
# The quick brown fox jumps over the lazy dog
# ['The quick brown fox jumps over the lazy dog']
# The quick brown fox jumps over the lazy dog
# ['The quick brown fox jumps over the lazy dog']
# This is a test
# ['This is a test']
# This is a test
# ['This is a test']
# This is a random test
# ['This is a ', ' a random test']
# The quick brown fox and a random test
# ['The quick brown fox ', ' a random test']
```

### Transformer Embedding Semantic Deduplication

```python
from text_dedup.embedders.transformer import TransformerEmbedder
from text_dedup.utils.nn import annoy_clustering
from text_dedup.utils.group import get_group_indices

from transformers import AutoTokenizer, AutoModelForSequenceClassification
corpus = [
    "The quick brown fox jumps over the dog",
    "The quick brown fox jumps over the corgi",
    "This is a test",
    "This is a test message",
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

embedder = TransformerEmbedder(tokenizer, model)
embeddings = embedder.embed(corpus)

clusters = annoy_clustering(embeddings, f=768)
groups = get_group_indices(clusters)
print(groups)
# [0, 0, 2, 2]
```

### Best Fuzzy Search

This is useful for ad-hoc fuzzy substring search. Given a long document and a query string, this function will return a best fuzzy match based on Jaccard similarity.

```python
from text_dedup.utils.search import best_fuzzy_search

best_fuzzy_search("Hello world!", "Random word, Hello word! hello menudo!")
# (13, 'Hello word!')
```

## Benchmarks

## Todos

-   [ ] Wrap suffix array inter-split deduplication
-   [ ] Wrap inter-dataset deduplication
-   [ ] Rewrite suffix array in Python

## Thanks

-   [seomoz/simhash-cpp](https://github.com/seomoz/simhash-cpp)
-   [datasketch](http://ekzhu.com/datasketch/index.html)
-   [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)

This project is heavily influenced by the deduplication work at BigScience workshop. The original code can be found at [bigscience-workshop/data-preparation](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/filtering/deduplicate).
