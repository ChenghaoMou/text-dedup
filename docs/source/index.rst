This is the documentation for `text_dedup` — a collection of Python scripts for text deduplication. Each script maintains a unified command line interface that takes in a huggingface dataset and produces a deduplicated version of it. Some scripts are designed to handle large datasets and should be run on a cluster. Others are designed to be run on a single machine for small to medium sized datasets.

API Reference
-------------
.. toctree::
   :maxdepth: 1

   minhash
   minhash_spark
   simhash
   suffix_array
   exact_hash
   ccnet
   utils



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
