This is the documentation for `text_dedup` — a collection of Python scripts for text deduplication. Each script maintains a unified command line interface that takes in a huggingface dataset and produces a deduplicated version of it. Some scripts are designed to handle large datasets and should be run on a cluster. Others are designed to be run on a single machine for small to medium sized datasets.

.. note::
   These scripts are meant to be modified to your specific use case. No settings mentioned in this documentation are universally applicable to every language and dataset. You will need to experiment with different data format, tokenization, thresholds, and other settings to get the best results for your use case. I will try my best to include my intuitions for why I chose certain settings and how they performed in my experiments, but you should not take my word for it.

   If you are interested in how these scripts were developed and how they performed for BigCode, please refer to the blog post: https://huggingface.co/blog/dedup.

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
