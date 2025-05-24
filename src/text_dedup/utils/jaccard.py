#
# created     : Thu May 22 2025
# author      : Chenghao Mou <mouchenghao at gmail dot com>
# license     : Apache-2.0
# description : Compute the Jaccard similarity between two set of tokens.
#


def jaccard_similarity(
    doc1: set[str],
    doc2: set[str],
) -> float:
    """Compute the Jaccard similarity between two set of tokens.

    Parameters
    ----------
    doc1 : set[str]
        The first set of tokens.
    doc2 : set[str]
        The second set of tokens.

    Returns
    -------
    float
        The Jaccard similarity.
    """
    return len(doc1 & doc2) / max(1, len(doc1 | doc2))
