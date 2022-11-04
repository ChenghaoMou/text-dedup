#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/1/22
# description :  Exact deduplication using bloom filter.
import logging
import os
from dataclasses import dataclass
from functools import partial
from hashlib import md5
from hashlib import sha256
from typing import Callable
from typing import List
from typing import Sequence

from pybloom import ScalableBloomFilter
from tqdm import tqdm

from text_dedup.exact_dedup.base import ExactDuplicateFinder
from text_dedup.preprocess import tokenize


@dataclass
class BloomFilter(ExactDuplicateFinder):

    """
    Bloom filter is a probabilistic data structure that is used to test whether an element is a member of a set.

    The false positive probability of a bloom filter is the probability that the bloom filter incorrectly
    identifies an element as being in the set when it is not. This is different from using a dictionary or hash table.

    Parameters
    ----------
    hash_func: Callable
        The hash function used to generate the hash value of the text.
    error_rate: float
        The false positive probability of the bloom filter.
    """

    hash_func: Callable = md5
    error_rate: float = 1e-6
    tokenizer: Callable[..., List[str]] = tokenize

    def __post_init__(self):
        self.bf = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH, error_rate=self.error_rate)

    def fit(self, data: Sequence[str]):
        for text in tqdm(data):
            self.bf.add(self.hash_func(" ".join(self.tokenizer(text)).encode("utf-8")).hexdigest())

    def fit_fingerprints(self, data: Sequence[str]):
        for h in tqdm(data):
            self.bf.add(h)

    def predict(self, data: Sequence[str]) -> List[bool]:
        return [self.hash_func(" ".join(self.tokenizer(text)).encode("utf-8")).hexdigest() not in self.bf for text in tqdm(data)]

    def predict_fingerprints(self, data: Sequence[str]) -> List[bool]:
        return [h not in self.bf for h in tqdm(data)]

    def fit_predict(self, data: Sequence[str]) -> List[bool]:
        mask = []
        for text in tqdm(data):
            if not self.bf.add(self.hash_func(" ".join(self.tokenizer(text)).encode("utf-8")).hexdigest()):
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def fit_predict_fingerprints(self, data: Sequence[str]) -> List[bool]:
        mask = []
        for h in tqdm(data):
            if not self.bf.add(h):
                mask.append(True)
            else:
                mask.append(False)
        return mask


if __name__ == "__main__":  # type: ignore

    import time
    from collections import Counter

    import typer
    from datasets import load_dataset

    logger = logging.getLogger("text_dedup")

    def run(
            dataset: str,
            config: str,
            split: str,
            hash_func: str = "md5",
            verbose: bool = False,
    ):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        start_time = time.time()
        ds = load_dataset(dataset, config, split=split, use_auth_token=True)
        deduplicator = BloomFilter(
            hash_func=md5 if hash_func == "md5" else sha256,
            tokenizer=partial(tokenize, n_gram=1, step_size=1, level="code")
        )
        ds = ds.map(lambda x: {
            "fingerprint": deduplicator.hash_func(" ".join(deduplicator.tokenizer(x["content"])).encode("utf-8")).hexdigest()
        }, num_proc=os.cpu_count())
        mask = deduplicator.fit_predict_fingerprints(ds["fingerprint"])
        logger.info(f"Duplicates: {Counter(mask)[False]}")
        logger.info(f"Finished in {time.time() - start_time:.2f} seconds.")

    typer.run(run)
