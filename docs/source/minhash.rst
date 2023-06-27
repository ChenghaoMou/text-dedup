MinHash + LSH
=============

This script implements a simple MinHash + LSH algorithm for finding near duplicates. This is largely based on the implementation from `datasketch <https://github.com/ekzhu/datasketch>`_ (MIT).

Quick Start
-----------

.. code-block:: bash

   usage: text_dedup.minhash [-h] --path PATH [--name NAME] [--data_dir DATA_DIR] [--data_files DATA_FILES]
                          [--split SPLIT] [--cache_dir CACHE_DIR] [--revision REVISION]
                          [--use_auth_token | --no-use_auth_token] [--local | --no-local] --output OUTPUT
                          [--debug | --no-debug] --column COLUMN [--batch_size BATCH_SIZE] [--ngram NGRAM]
                          [--min_length MIN_LENGTH] [--seed SEED] [--num_perm NUM_PERM] [--threshold THRESHOLD]
                          [--b B] [--r R]

Deduplicate text using minhash

options:
  -h, --help            show this help message and exit
  --path PATH           `path` in load_dataset
  --name NAME           `name` in load_dataset
  --data_dir DATA_DIR   `data_dir` in load_dataset
  --data_files DATA_FILES
                        `data_files` in load_dataset
  --split SPLIT         `split` in load_dataset
  --cache_dir CACHE_DIR
                        `cache_dir` in load_dataset
  --revision REVISION   `revision` in load_dataset
  --use_auth_token, --no-use_auth_token
                        `use_auth_token` in load_dataset
  --local, --no-local   Use local dataset (default: False)
  --output OUTPUT       Path to deduplicated dataset output
  --debug, --no-debug   Whether to run in debug mode (default: False)
  --column COLUMN       Text column to use for deduplication. Concatenate desired columns beforehand if needed.
  --batch_size BATCH_SIZE
                        Batch size to use for dataset iteration. Mainly for memory efficiency.
  --ngram NGRAM         Ngram size to use in MinHash.
  --min_length MIN_LENGTH
                        Minimum number of tokens to use in MinHash. Shorter documents will be filtered out.
  --seed SEED           Seed to use in MinHash
  --num_perm NUM_PERM   Number of permutations to use in MinHash
  --threshold THRESHOLD
                        Jaccard similarity threshold to use in MinHashLSH
  --b B                 Number of bands
  --r R                 Number of rows per band

Example
-------

.. code-block:: bash

   python -m text_dedup.minhash \
      --path "oscar-corpus/OSCAR-2201" \
      --name "gl" \
      --split "train" \
      --cache_dir "./cache" \
      --output "output/minhash/oscar_gl_dedup" \
      --column "text" \
      --batch_size 10000

API Reference
-------------
.. automodule:: text_dedup.minhash
   :members:
   :undoc-members:
   :noindex: