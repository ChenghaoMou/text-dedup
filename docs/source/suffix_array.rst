Suffix Array Substring
======================

.. warning::
    Currently, there is an issue with merge command from the `original repo <https://github.com/google-research/deduplicate-text-datasets/issues/19>`_, which might cause the processing to be single-threaded. You can apply `this fix <https://github.com/google-research/deduplicate-text-datasets/pull/22>`_ to the original repo to fix the issue.

This is a wrapper around `deduplicate-text-datasets <https://github.com/google-research/deduplicate-text-datasets>`_ to deduplicate text datasets using suffix array substring matching. Based on the recommendation from the original research, duplicated substrings will be removed from the dataset. 

    *"In our paper we suggest just taking all of these duplicate sequences that have been identified and completely striking them from the dataset. This somewhat breaks the flow of text, for example if previously had an example "Alice wanted to go to the store" and we deduplicated at the level of 10 characters, we might completely strike " to go to the " and be left with "Alice wantedstore". In practice we have found this doesn't break the language model because we remove relatively little text, and so these breaks don't cause harm."*

This wrapper adds one more step to the original script by restoring the text back into their document boundaries. This way, you still get the original documents, but with the duplicated substrings removed, instead of one long string of text. However, this boundary-respecting step is not perfect, and might not remove all the byte sequence since the original script yields byte offsets and normal text uses unicode characters. In this case, erroneous bytes around the offsets or boundaries will be ignored.

Quick Start
-----------
.. code-block:: bash

   python -m text_dedup.suffix_array \
    --path "oscar-corpus/OSCAR-2201" \
    --name "gl" \
    --split "train" \
    --cache_dir "./cache" \
    --output "output" \
    --column "text" \
    --google_repo_path "deduplicate-text-datasets"


API Reference
-------------
.. automodule:: text_dedup.suffix_array
   :members:
   :undoc-members:
   :noindex:

