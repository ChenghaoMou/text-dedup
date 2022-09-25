CLI References
==============

Finding duplicates in a Huggingface dataset
-------------------------------------------

``cli.py`` is a wrapper tool that identifies duplicates for a given Huggingface's dataset.

By default, the tool uses ``redis`` or ``key-db`` as a cache layer for the hashes. See ``configs/method/minhash.yaml`` or ``configs/method/simhash.yaml`` for examples. Or you can overwrite the `storage_config` to `null` to use in-memory index. Deduplicating small datasets that fit in your machine's memory should be fine with in-memory index.

.. warning::

  When using ``MinHashEmbedder``, please use redis server instead of key-db. Key-db is not compatible yet with some data types.

Configuration
-------------

The configuration file is a YAML file that contains the following fields:

- ``dataset``: the name of the dataset to be analyzed. It must be a valid Huggingface's dataset name.
- ``configs``: dataset configs to use. It must be a list of configs for the dataset. For example, for the ``oscar-corpus/OSCAR-2109`` dataset, you can use ``deduplicated_gl``.
- ``columns``: the columns/features to be used to form the final text.
- ``num_proc``: the number of processes to use for the analysis. Use ``1`` for streaming processing.
- ``embedder``: configuration for the fingerprinting method.
- ``tokenization``: configuration for the tokenization method.
- ``storage_config``: configuration for redis storage. It can be ``null`` to use in-memory index.
- ``cache_dir``: the directory to store the cache files.
- ``output``: the output file to store the results.
- ``verbose``: whether to print the progress or not.

Embedder Configuration
----------------------

The ``embedder`` field is a dictionary that contains the following fields depending on the fingerprinting method:

MinHashEmbedder
~~~~~~~~~~~~~~~

::

    embedder:
        name: MinHashEmbedder
        num_perm: 128
        threshold: 0.8

- ``embedder.name``: the name of the fingerprinting method. It must be ``MinHashEmbedder``.
- ``embedder.num_perm``: the number of permutations to use for the MinHash algorithm.
- ``embedder.threshold``: the threshold to use for the MinHash algorithm.

SimHashEmbedder
~~~~~~~~~~~~~~~

::

    embedder:
        name: SimHashEmbedder
        hamming_distance: 3
        num_blocks: 4

- ``embedder.name``: the name of the fingerprinting method. It must be ``SimHashEmbedder``.
- ``embedder.hamming_distance``: the bit difference threshold to use for the SimHash algorithm.
- ``embedder.num_blocks``: the number of blocks to use for the SimHash algorithm.

SuffixArrayEmbedder
~~~~~~~~~~~~~~~~~~~

::

    embedder:
      name: SuffixArrayEmbedder
      k: 50
      skip_existing: true
      cache_dir: outputs
      temp_file_prefix: sa


- ``embedder.name``: the name of the fingerprinting method. It must be ``SuffixArrayEmbedder``.
- ``embedder.k``: the byte length threshold to use for the Suffix Array algorithm.
- ``embedder.skip_existing``: skip generating files if they already exist.
- ``embedder.cache_dir``: the directory to store the cache files for the Suffix Array algorithm.
- ``embedder.temp_file_prefix``: the prefix of the temporary files for the Suffix Array algorithm.

Tokenization Configuration
--------------------------

The ``tokenization`` field is a dictionary that contains the following fields depending on the tokenization method:

- ``tokenization.ngram_size``: the size of the n-gram to use for the tokenization.
- ``tokenization.level``: the level of the tokenization. It can be ``word``, ``char``, or ``sentencepiece``. When using ``sentencepiece``, it uses ``XLMRobertaTokenizerFast``.

Storage Configuration
---------------------

The ``storage_config`` field is a dictionary that contains the following fields:

- ``storage_config.type``: the type of the storage. It can only be ``redis`` for now.
- ``storage_config.redis.host``: the host of the storage.
- ``storage_config.redis.port``: the port of the storage.

You can also use ``null`` to use in-memory index.

::

    storage_config: null

    # or
    storage_config:
      type: redis
      redis:
        host: localhost
        port: 6379

Examples
--------

deduplicate the oscar-corpus/OSCAR-2109 dataset using simhash with configs from configs/method/simhash.yaml and some overrides

::

    # make sure to start key-db server first or modify the config first
    python cli.py method=simhash method.dataset=oscar-corpus/OSCAR-2109 method.configs="[deduplicated_gl]"

deduplicate the oscar-corpus/OSCAR-2109 dataset using minhash with configs from configs/method/minhash.yaml and some overrides

::

    # make sure to start redis server first or modify the config first
    python cli.py method=minhash method.dataset=oscar-corpus/OSCAR-2109 method.configs="[deduplicated_gl]"

deduplicate the oscar-corpus/OSCAR-2109 dataset using suffix array with configs from configs/method/suffix.yaml and some overrides

::

    python cli.py method=suffix method.dataset=oscar-corpus/OSCAR-2109 method.configs="[deduplicated_gl]"
