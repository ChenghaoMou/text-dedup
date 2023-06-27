SimHash
=======

This script is a simple implementation of the SimHash algorithm.

Quick Start
-----------

 .. code-block:: bash

   usage: text_dedup.simhash [-h] --path PATH [--name NAME] [--data_dir DATA_DIR] [--data_files DATA_FILES]
                          [--split SPLIT] [--cache_dir CACHE_DIR] [--revision REVISION]
                          [--use_auth_token | --no-use_auth_token] [--local | --no-local] --output OUTPUT
                          [--debug | --no-debug] --column COLUMN [--batch_size BATCH_SIZE] [--ngram NGRAM]
                          [--f F] [--bit_diff BIT_DIFF] [--num_bucket NUM_BUCKET]

Deduplicate text using simhash

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
  --ngram NGRAM         Ngram size to use in SimHash.
  --f F                 Simhash bit size
  --bit_diff BIT_DIFF   Bit difference to use in SimHash
  --num_bucket NUM_BUCKET
                        Number of buckets to use in SimHash, must be larger than bit_diff

Example
-------

.. code-block:: bash

  python -m text_dedup.simhash \
  --path "oscar-corpus/OSCAR-2201" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/simhash/oscar_gl_dedup" \
  --column "text" \
  --batch_size 10000

API Reference
-------------

.. automodule:: text_dedup.simhash
   :members:
   :undoc-members:
   :noindex:
