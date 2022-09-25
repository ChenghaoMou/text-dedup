.. text-dedup documentation master file, created by
   sphinx-quickstart on Wed Jul 13 19:54:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

text-dedup
======================================

Deduplication is a common task in data processing, however, it is not always easy to do for large text datasets. We already have frameworks for models_, datasets_, and even metrics_. But data analysis and processing is still largely under-developed. Of course, this is not my ambitious declaration of conquering those problems but rather a way to contribute to the community.

I started this as a side project and later on I joined the `Big Science`_ initiative to work on data deduplication. Now I am putting what I have learned from that experience into this package and hopefully it will be useful to you as well.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   examples
   text_dedup
   results

Concepts
--------

Deduplication itself is easy to understand. It is a process of identifying duplicate records. Historically, there are two forms of deduplication, namely, **exact deduplication** and **near deduplication**. The exact deduplication is the most common and is based on the fact that two records are identical if they have the same text value. The near deduplication is a bit more complicated as similarity itself is a spectrum. Text can be almost identical to each other having few characters different to somewhat similar having the same template. Due to the recent advancement on representation learning and large pre-trained language models, now we can also have **semantic deduplication** where text can be considered as duplicates if they refer to the same thing, e.g. different report on the same event. Depending on your use case and constraints, you can also have different flavors of deduplication within each category.

But in general, there is a nice abstraction we can use to summarize the above-mentioned deduplication techniques, which is used in most Natural Language Processing (NLP) tasks — representation then action.

Representation
--------------

Embedding, fingerprint, representation, or encoding are often used interchangeably in this case. In terms of deduplication, the reason why we need a mathematical representation of the data since we already have the text is to identify duplicate records more EFFICIENTLY.

This library primarily uses the words ``embedding`` and ``embedders`` to describe the process and the word ``fingerprint`` to describe the outcome.

Action
------

Once you have the representation, you can use it to identify duplicate records. However, this can be a bit tricky to do for near deduplication. For example, if document A is similar to document B, and document B is also similar to document C, but document A may or may not be similar to document C. For another example, a document might share some sub-string with another document, do you want to remove the sub-string and therefore breaking the text flow or you want to keep the duplicate and not to have text fragments?


Considerations
--------------

There are many things to consider when it comes to data, so here I outline some of the most important ones:

- Language agnostic.
   - If there are tokenization involved, at least use a multi-lingual tokenizer.
   - If there are any special characters or configurations, be transparent.
- Efficiency and Scalability.
   - Fast for small to medium datasets.
   - Memory efficient, especially for large datasets (>TB).
- Transparency and flexibility.
   - Configurations for reproducibility.
   - Intra- and inter- dataset/splits deduplication.
   - Customization for different features.
   - Easy to use without the need of deep understanding.
- Interpretability.
   - Easy to understand the results.
   - Easy to import and export the results.


Feedback, Issues, and Contributions
===================================

Please consider opening an issue or a pull request if you have any questions, comments, or suggestions at Github_.

.. _models: https://github.com/huggingface/transformers
.. _datasets: https://github.com/huggingface/datasets
.. _metrics: https://github.com/huggingface/evaluate
.. _Big Science: https://github.com/bigscience-workshop/data_tooling
.. _Github: https://github.com/ChenghaoMou/text-dedup

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
