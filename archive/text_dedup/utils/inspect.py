import random
from collections import defaultdict

from datasets import Dataset

from text_dedup import logger


def random_samples(
    ds: Dataset,
    cluster_column: str,
    text_column: str,
    num_clusters: int = 10,
    num_examples_per_cluster: int = 5,
):
    cluster2idx = defaultdict(set)
    for idx, cluster in enumerate(ds[cluster_column]):
        cluster2idx[cluster].add(idx)

    candidates = [key for key in cluster2idx if len(cluster2idx[key]) > 1]
    clusters = random.sample(candidates, min(num_clusters, len(candidates)))
    for cluster in clusters:
        logger.info(f"Cluster: {cluster}")
        for idx in list(cluster2idx[cluster])[:num_examples_per_cluster]:
            logger.info(f"#{idx}: {ds[idx][text_column]}")
