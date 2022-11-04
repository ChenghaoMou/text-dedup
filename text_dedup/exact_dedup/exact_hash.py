#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/30/22
# description : A modified version of the pile's exact deduplication code. (MIT)
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from hashlib import sha256
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence

from tqdm import tqdm

from text_dedup.exact_dedup.base import ExactDuplicateFinder
from text_dedup.preprocess import tokenize

logger = logging.getLogger("text_dedup")


@dataclass
class ExactHash(ExactDuplicateFinder):  # type: ignore

    h: Callable[[str], str] = lambda x: sha256(x.encode("utf-8")).hexdigest()
    keep: Literal["first", "last"] = "first"
    hash2idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    tokenizer: Callable[..., List[str]] = tokenize

    def fit(self, data: Sequence[str]):
        for i, text in enumerate(tqdm(data)):
            fingerprint = self.h(" ".join(self.tokenizer(text)))
            if fingerprint not in self.hash2idx:
                self.hash2idx[fingerprint] = i
            else:
                self.hash2idx[fingerprint] = min(self.hash2idx[fingerprint], i) if self.keep == "first" else max(
                    self.hash2idx[fingerprint], i
                )

    def fit_fingerprints(self, data: Sequence[str]):
        for i, h in enumerate(tqdm(data)):
            if h not in self.hash2idx:
                self.hash2idx[h] = i
            else:
                self.hash2idx[h] = min(self.hash2idx[h], i) if self.keep == "first" else max(self.hash2idx[h], i)

    def predict(self, data: Sequence[str]) -> List[bool]:
        return [sha256(" ".join(self.tokenizer(text)).encode("utf-8")).hexdigest() in self.hash2idx for text in tqdm(data)]

    def predict_fingerprints(self, data: Sequence[str]) -> List[bool]:
        return [h in self.hash2idx for h in tqdm(data)]

    def fit_predict(self, data: Sequence[str]) -> List[bool]:
        mask = [False for _ in range(len(data))]
        for i, text in enumerate(tqdm(data)):
            fingerprint = sha256(" ".join(self.tokenizer(text)).encode("utf-8")).hexdigest()
            if fingerprint not in self.hash2idx:
                self.hash2idx[fingerprint] = i
                mask[i] = True if self.keep == "first" else False
            else:
                self.hash2idx[fingerprint] = min(self.hash2idx[fingerprint], i) if self.keep == "first" else max(
                    self.hash2idx[fingerprint], i
                )

        for fp in self.hash2idx:
            mask[self.hash2idx[fp]] = True

        return mask

    def fit_predict_fingerprints(self, data: Sequence[str]) -> List[bool]:
        mask = [False for _ in range(len(data))]
        for i, h in enumerate(tqdm(data)):
            if h not in self.hash2idx:
                self.hash2idx[h] = i
                mask[i] = True if self.keep == "first" else False
            else:
                self.hash2idx[h] = min(self.hash2idx[h], i) if self.keep == "first" else max(self.hash2idx[h], i)

        for fp in self.hash2idx:
            mask[self.hash2idx[fp]] = True

        return mask


if __name__ == "__main__":  # type: ignore

    import typer
    from datasets import load_dataset

    def run(
            dataset: str,
            config: str,
            split: str,
            keep: str = "first",  # typer does not support Literal yet
            verbose: bool = False,
    ):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        start_time = time.time()
        deduplicator = ExactHash(
            keep=keep,  # type: ignore
            tokenizer=partial(tokenize, n_gram=1, step_size=1, level="code")
        )
        ds = load_dataset(dataset, config, split=split, use_auth_token=True)
        ds = ds.map(
            lambda x: {
                "fingerprint": sha256(" ".join(deduplicator.tokenizer(x["content"])).encode("utf-8")).hexdigest()
            },
            num_proc=os.cpu_count()
        )
        mask = deduplicator.fit_predict_fingerprints(ds["fingerprint"])
        logger.info(f"Duplicates: {Counter(mask)[False]}")
        logger.info(f"Finished in {time.time() - start_time:.2f} seconds.")

    typer.run(run)
