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

Intuitions on Parameters
------------------------

In this section, I'll cover the following parameters based on my own experience:

- `ngram`
- `num_perm`
- `threshold`
- `b` and `r`

An interactive demo can be found `here <https://huggingface.co/spaces/bigcode/near-deduplication>`_.

`ngram` or tokenization in general is a way of describing the document. After all, MinHash is an approximation of the Jaccard similarity between two sets — two sets of ngrams in this case. The quality of the ngrams will impact your final results. For example, if you use character uni-gram, you will almost certainly get a lot of false positives for English text or DNA sequences. But it won't necessarily be a problem for CJK data. I usually start with word-level tri-grams and adjust from there. Strangely enough, people almost always choose odd numbers for `ngram` (e.g. 3, 5, 7, etc.). I don't know why, but I do it too.

`num_perm` is the number of permutations to use in MinHash. It is tightly related to the band and row numbers. The higher the permutation number, the more accurate the Jaccard similarity estimation will be. But it will also be slower and use more space (the space complexity is O(num_perm)). I usually start with something like 128 or 256 and adjust from there. One thing you could try is to play with the interactive demo to see how much false positive and false negative you are willing to tolerate, and then adjust those settings accordingly. 

You might see something like 9000 permutations used in research papers :cite:p:`lee2022deduplicating`, along with additional second-stage filtering (e.g. edit similarity) to reduce the false positives. If you modify the interactive demo code to select 9k permutations, 450 bands, and 20 rows per band, you will see that you will more likely to expect false positives, which necessitates the second-stage filtering. Of course you can choose something different when you have different priorities.

Time and Space Complexity
-------------------------

The time complexity of MinHash is O(num_docs * doc_length * num_perm / num_proc). The space complexity is O(num_docs * num_perm).

The time complexity of LSH is O(num_docs * b). The space complexity varies depending on how many duplicates are there. If all documents are unique, you would expect O(num_docs * b). If all documents are duplicates, you would expect O(b).

If you add secondary filtering in the process, the time complexity will be higher.

.. bibliography:: refs.bib
   :style: unsrt
   :filter: docname in docnames

API Reference
-------------
.. automodule:: text_dedup.minhash
   :members:
   :undoc-members:
   :noindex: