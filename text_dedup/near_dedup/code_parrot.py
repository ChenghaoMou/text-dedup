#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/30/22
# description : A modified version of code parrot's near deduplication code. (Apache 2.0)

"""
examples/research_projects/codeparrot/scripts/minhash_deduplication.py
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import re
import time
from collections import defaultdict
from functools import partial
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple

from datasets import Dataset
from datasets import load_dataset
from datasketch import MinHash
from datasketch import MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator
from tqdm import tqdm

logger = logging.getLogger("text_dedup")
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MIN_NUM_TOKENS = 10
NUM_PERM = 256

CONTENT_KEY = "content"
REPO_KEY = "repo_name"
PATH_KEY = "path"

CodeKey = NamedTuple("CodeKey", [("key", str), ("repo_name", str), ("path", str)])


def get_min_hash(tokens: Set[str]) -> Optional[MinHash]:
    """Compute the MinHash of a code snippet."""
    if len(tokens) < MIN_NUM_TOKENS:
        return None
    min_hash = MinHash(num_perm=NUM_PERM)
    min_hash.update_batch([token.encode() for token in tokens])
    return min_hash


def get_tokens(code: str) -> Set[str]:
    """Tokenize a code snippet."""
    return set([t for t in NON_ALPHA.split(code) if t.strip()])


class CodeParrotDuplicationIndex:
    def __init__(
            self,
            num_perm: int = NUM_PERM,
            duplication_jaccard_threshold: float = 0.85,
    ):
        self._duplication_jaccard_threshold = duplication_jaccard_threshold
        self._num_perm = num_perm
        self._index = MinHashLSH(threshold=self._duplication_jaccard_threshold, num_perm=self._num_perm)
        self._duplicate_clusters: Dict[Any, Set] = defaultdict(set)

    def add(self, code_key: CodeKey, min_hash: MinHash) -> None:

        if code_key in self._index.keys:
            logger.debug(f"Duplicate key {code_key}")
            return

        # IF B is added to A's cluster
        # Then C is added to B's cluster when C is slightly more different from A.
        # Both A and B can be kept when they are the "extreme"s in each cluster.

        close_duplicates = self._index.query(min_hash)
        self._index.insert(code_key, min_hash)

        if len(close_duplicates) <= 0:
            return

        for base_duplicate in close_duplicates:
            if base_duplicate in self._duplicate_clusters:
                self._duplicate_clusters[base_duplicate].add(code_key)
                return

        self._duplicate_clusters[close_duplicates[0]].add(code_key)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        duplicate_clusters: List[List[Dict]] = []
        for base, duplicates in self._duplicate_clusters.items():
            cluster = [base] + list(duplicates)
            cluster = [{"base_index": el.key, "repo_name": el.repo_name, "path": el.path} for el in cluster]
            duplicate_clusters.append(cluster)
        return duplicate_clusters

    def save(self, filepath) -> None:
        duplicate_clusters = self.get_duplicate_clusters()
        with open(filepath, "w") as f:
            json.dump(duplicate_clusters, f)


def _compute_min_hash(element):
    index, data = element
    min_hash = get_min_hash(get_tokens(data[CONTENT_KEY]))
    if min_hash is not None:
        return CodeKey(index, data[REPO_KEY], data[PATH_KEY]), min_hash


def minhash_iter(dataset_iterator: Iterator):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
                _compute_min_hash,
                ThreadedIterator(dataset_iterator, max_queue_size=10000),
                chunksize=100,
        ):
            if data is not None:
                yield data


def make_duplicate_clusters(
        dataset_iterator: Dataset | Iterator,
        num_perm: int = NUM_PERM,
        jaccard_threshold: float = 0.85
):
    di = CodeParrotDuplicationIndex(num_perm=num_perm, duplication_jaccard_threshold=jaccard_threshold)

    for filename, min_hash in tqdm(
            ThreadedIterator(minhash_iter(enumerate(dataset_iterator)), max_queue_size=100),
            desc="Building index"
    ):
        di.add(filename, min_hash)

    return di.get_duplicate_clusters()


def jaccard_similarity(code1: str, code2: str) -> float:
    """Compute the Jaccard similarity of two code snippets."""
    tokens1 = get_tokens(code1)
    tokens2 = get_tokens(code2)
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def _find_cluster_extremes_shared(cluster: List[Dict], jaccard_threshold: float):
    extremes: List[Dict] = []
    for element1 in cluster:
        code1 = element1["content"]
        for element2 in extremes:
            code2 = element2["content"]
            if jaccard_similarity(code1, code2) >= jaccard_threshold:
                element2["copies"] += 1
                break
        else:
            element1["copies"] = 1
            extremes.append(element1)
    return extremes


def find_extremes(cluster_list, dataset: Dataset, jaccard_threshold: float = 0.85):
    # @ChenghaoMou Assuming duplicates can be fitted in memory
    cluster_list = [[element | {"content": dataset[element["base_index"]]["content"]} for element in cluster] for
                    cluster in cluster_list]
    extremes_list = []
    f = partial(_find_cluster_extremes_shared, jaccard_threshold=jaccard_threshold)
    with mp.Pool() as pool:
        for extremes in tqdm(
                pool.imap_unordered(
                    f,
                    cluster_list,
                ),
                total=len(cluster_list),
                desc="Finding extremes",
        ):
            extremes_list.append(extremes)
    return extremes_list


def deduplicate_dataset(
        dataset: Dataset, num_perm: int = NUM_PERM, jaccard_threshold: float = 0.85
) -> Tuple[Dataset, List[List[Dict]]]:
    duplicate_clusters = make_duplicate_clusters(dataset, num_perm=num_perm, jaccard_threshold=jaccard_threshold)
    duplicate_indices = set(x["base_index"] for cluster in duplicate_clusters for x in cluster)
    extreme_dict = {}
    extremes_clusters = find_extremes(duplicate_clusters, dataset, jaccard_threshold)
    for extremes in extremes_clusters:
        for element in extremes:
            extreme_dict[element["base_index"]] = element
    remove_indices = duplicate_indices - set(extreme_dict.keys())
    ds_filter = dataset.filter(lambda x, idx: idx not in remove_indices, with_indices=True)

    # update duplicate_clusters
    for cluster in duplicate_clusters:
        for element in cluster:
            element["is_extreme"] = element["base_index"] in extreme_dict
            if element["is_extreme"]:
                element["copies"] = extreme_dict[element["base_index"]]["copies"]

    logger.info(f"Original dataset size: {len(dataset)}")
    logger.info(f"Number of duplicate clusters: {len(duplicate_clusters)}")
    logger.info(f"Files in duplicate cluster: {len(duplicate_indices)}")
    logger.info(f"Unique files in duplicate cluster: {len(extreme_dict)}")
    logger.info(f"Filtered dataset size: {len(ds_filter)}")

    return ds_filter, duplicate_clusters


if __name__ == "__main__":

    import typer

    def run(
            dataset: str,
            config: str,
            split: str,
            num_perm: int = NUM_PERM,
            jaccard_threshold: float = 0.85,
            verbose: bool = False
    ):
        start_time = time.time()
        if verbose:
            # set the logging level to the lowest
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose mode activated")
        ds = load_dataset(dataset, config, split=split, use_auth_token=True)
        deduplicate_dataset(ds, num_perm=num_perm, jaccard_threshold=jaccard_threshold)
        logger.info(f"Total time: {time.time() - start_time}")
        pass

    typer.run(run)
