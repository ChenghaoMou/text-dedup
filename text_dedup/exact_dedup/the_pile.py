#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/30/22
# description : A modified version of the pile's exact deduplication code. (MIT)
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from hashlib import sha256
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence
from typing import Set

logger = logging.getLogger("text_dedup")


@dataclass
class ThePileExactDeduplicator:  # type: ignore
    keep: Literal["first", "last"] = "first"
    _hashes: Dict[str, Set[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._hashes = defaultdict(set)

    def fit(self, data: Sequence[str]):
        for i, text in enumerate(data):
            fingerprint = sha256(text.encode("utf-8")).hexdigest()
            self._hashes[fingerprint].add(i)

    def predict(self, data: Sequence[str]) -> List[Set[int]]:
        return [self._hashes[sha256(text.encode("utf-8")).hexdigest()] for text in data]

    def fit_predict(self, data: Sequence[str]) -> List[Set[int]]:
        self.fit(data)
        return self.predict(data)

    def drop_duplicates(self, ds, num_proc: int = 1):
        ids = set(list(range(len(ds))))
        to_keep = set()
        for group in self._hashes.values():
            if self.keep == "first":
                to_keep.add(min(group))
            else:
                to_keep.add(max(group))

        ds = ds.filter(lambda _, idx: idx in to_keep, with_indices=True, num_proc=num_proc)

        logger.info(f"Removed {len(ids) - len(to_keep)} duplicates.")

        return ds


if __name__ == "__main__":  # type: ignore

    import typer
    from datasets import load_dataset

    def run(
            dataset: str,
            config: str,
            split: str,
            keep: str = "first",
            verbose: bool = False,
    ):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        start_time = time.time()
        ds = load_dataset(dataset, config, split=split, use_auth_token=True)
        deduplicator = ThePileExactDeduplicator(keep=keep)  # type: ignore
        deduplicator.fit(ds["content"])
        deduplicator.drop_duplicates(ds, num_proc=1)
        logger.info(f"Finished in {time.time() - start_time:.2f} seconds.")

    typer.run(run)
