CLI References
==============

Finding duplicates in a Huggingface dataset
-------------------------------------------

`cli.py` is a wrapper tool that identifies duplicates for a given Huggingface's dataset. Currently, only hash-based methods will try to identify all duplicates within the dataset and the suffix array method will only find the duplicate substrings within dataset splits.

By default, the tool uses redis as a cache layer for the hashes. See `configs/method/minhash.yaml` or `configs/method/simhash.yaml` for details. Or you can overwrite the `storage_config` to `null` to use in-memory index. Deduplicating small datasets that fit in your machine's memory should be fine with in-memory index.

```text
python cli.py method=suffix  method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
python cli.py method=simhash method.tokenization.ngram_size=12 method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
python cli.py method=minhash method.tokenization.ngram_size=12 method.dataset=oscar-corpus/OSCAR-2201 method.configs="[gl]"
```

-   Configurations are parsed with [hydra](https://hydra.cc).
