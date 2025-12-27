#
# created     : Thu May 22 2025
# author      : Chenghao Mou <mouchenghao at gmail dot com>
# license     : Apache-2.0
# description : Compute the Jaccard similarity between two set of tokens.
#


def jaccard_similarity(
    doc1: set[str] | set[bytes],
    doc2: set[str] | set[bytes],
) -> float:
    """Compute the Jaccard similarity between two set of tokens.

    Parameters
    ----------
    doc1 : set[str]|set[bytes]
        The first set of tokens.
    doc2 : set[str]|set[bytes]
        The second set of tokens.

    Returns
    -------
    float
        The Jaccard similarity.
    """
    if (union_size := len(doc1 | doc2)) == 0:
        return 1.0

    return len(doc1 & doc2) / union_size


def cluster_jaccard_similarity(
    cluster: list[set[bytes]],
    threshold: float,
) -> tuple[list[float], float]:
    if len(cluster) <= 1:
        return [], 0
    similarities: list[float] = []
    total = len(cluster)
    fp = 0
    for i, doc1 in enumerate(cluster):
        dup_similarity = max(jaccard_similarity(doc1, doc2) for j, doc2 in enumerate(cluster) if j != i)
        similarities.append(dup_similarity)
        if dup_similarity < threshold:
            fp += 1
    return similarities, fp / total
