"""Utility functions for benchmark evaluation."""

from collections import defaultdict


def precision(duplicates: set, predictions: set) -> float:
    """Calculate precision for duplicate detection.

    Parameters
    ----------
    duplicates : set
        Ground truth duplicates
    predictions : set
        Predicted duplicates

    Returns
    -------
    float
        Precision score
    """
    if len(predictions) == 0:
        return 0.0
    return len(duplicates & predictions) / len(predictions)


def recall(duplicates: set, predictions: set) -> float:
    """Calculate recall for duplicate detection.

    Parameters
    ----------
    duplicates : set
        Ground truth duplicates
    predictions : set
        Predicted duplicates

    Returns
    -------
    float
        Recall score
    """
    if len(duplicates) == 0:
        return 1.0
    return len(duplicates & predictions) / len(duplicates)


def f1_score(precision_val: float, recall_val: float) -> float:
    """Calculate F1 score.

    Parameters
    ----------
    precision_val : float
        Precision value
    recall_val : float
        Recall value

    Returns
    -------
    float
        F1 score
    """
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def classify_prediction(duplicates: set, predictions: set) -> str:
    """Classify a prediction as TP, FP, TN, or FN.

    Parameters
    ----------
    duplicates : set
        Ground truth duplicates
    predictions : set
        Predicted duplicates

    Returns
    -------
    str
        Classification label: TP, FP, TN, or FN
    """
    len_predictions = len(predictions)
    len_duplicates = len(duplicates)

    if len_predictions == 0:
        return "TN" if len_duplicates == 0 else "FN"

    if len_predictions > 0:
        if len_duplicates > 0 and duplicates.issubset(predictions):
            return "TP"
        return "FP"

    raise ValueError(f"Unexpected state: duplicates={duplicates}, predictions={predictions}")  # noqa: TRY003


def clusters_to_predictions_minhash(
    cluster_mapping: dict[int, int], id_to_core_id: dict[int, str]
) -> dict[str, set[str]]:
    """Convert MinHash cluster mapping to predictions dictionary.

    MinHash produces {document_id: cluster_group_id} where cluster_group_id
    is an arbitrary identifier, NOT a document ID.

    Parameters
    ----------
    cluster_mapping : dict[int, int]
        Mapping from document index to cluster group ID
    id_to_core_id : dict[int, str]
        Mapping from internal index to core_id

    Returns
    -------
    dict[str, set[str]]
        Mapping from core_id to set of duplicate core_ids
    """
    # Group documents by cluster
    cluster_to_docs: dict[int, set[str]] = defaultdict(set)
    for doc_idx, cluster_id in cluster_mapping.items():
        core_id = id_to_core_id.get(doc_idx)
        if core_id:
            cluster_to_docs[cluster_id].add(core_id)

    # For each document, find all other documents in the same cluster
    predictions: dict[str, set[str]] = {}
    for doc_idx, cluster_id in cluster_mapping.items():
        core_id = id_to_core_id.get(doc_idx)
        if core_id:
            cluster_docs = cluster_to_docs[cluster_id]
            predictions[core_id] = cluster_docs - {core_id}

    return predictions


def clusters_to_predictions_simhash(
    cluster_mapping: dict[int, int], id_to_core_id: dict[int, str]
) -> dict[str, set[str]]:
    """Convert SimHash cluster mapping to predictions dictionary.

    SimHash (union-find) produces {child_idx: parent_idx} where parent_idx
    IS a document ID (the cluster representative). Only children are in the
    mapping (parent != child).

    Parameters
    ----------
    cluster_mapping : dict[int, int]
        Mapping from child document index to parent document index
    id_to_core_id : dict[int, str]
        Mapping from internal index to core_id

    Returns
    -------
    dict[str, set[str]]
        Mapping from core_id to set of duplicate core_ids
    """
    cluster_to_docs: dict[str, set[str]] = defaultdict(set)

    for child_idx, parent_idx in cluster_mapping.items():
        child_core_id = id_to_core_id.get(child_idx)
        parent_core_id = id_to_core_id.get(parent_idx)

        if child_core_id and parent_core_id:
            # Use parent_core_id as the cluster key
            # Add both child and parent to the cluster
            cluster_to_docs[parent_core_id].add(child_core_id)
            cluster_to_docs[parent_core_id].add(parent_core_id)

    predictions: dict[str, set[str]] = {}
    for _, members in cluster_to_docs.items():
        for doc_id in members:
            predictions[doc_id] = members - {doc_id}

    return predictions
