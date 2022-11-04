import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from datasketch import LeanMinHash
from datasketch import MinHashLSH
from mpire import WorkerPool as Pool

from text_dedup.base import DuplicateFinder
from text_dedup.base import Embedder
from text_dedup.base import Fingerprint
from text_dedup.near_dedup.minhash.minhash_embedder import MinHashEmbedder
from text_dedup.preprocess import tokenize


@dataclass
class MinHashDeduplicator(DuplicateFinder):
    """
    Deduplicator using MinHash and MinHashLSH from datasketch.

    Parameters
    ----------
    seed: int
        The seed for the MinHash.
    threshold: float
        The threshold for the jaccard similarity, by default 0.8.
    num_perm: int
        The number of permutations for the MinHash, by default 128.
    storage_config: Optional[Dict[str, Any]]
        The storage config for the MinHashLSH, by default None.
    tokenizer: Callable[[str], List[str]]
        The tokenizer function, by default tokenize.
    verbose: bool
        Whether to show progress bar, by default False.
    """
    seed: int
    threshold: float = 0.8
    num_perm: int = 128
    storage_config: Optional[Dict[str, Any]] = None
    tokenizer: Callable[[str], List[str]] = tokenize
    verbose: bool = False
    __embedder: Embedder = field(init=False, repr=False)
    __index: MinHashLSH = field(init=False, repr=False)

    def __post_init__(self):
        self.__embedder = MinHashEmbedder(
            num_perm=self.num_perm, seed=self.seed, tokenizer=self.tokenizer)

    def fit(self, data: Sequence[str], **kwargs) -> 'MinHashDeduplicator':
        """
        Convert the data into fingerprints and index them.

        Parameters
        ----------
        data: Sequence[str]
            The data to be indexed.
        kwargs: Any
            The keyword arguments for the tokenizer function.

        Returns
        -------
        MinHashDeduplicator
            The fitted deduplicator.
        """
        signatures = self.transform(data, **kwargs)
        self.index(signatures)
        return self

    def predict(self, data: Sequence[str], **kwargs) -> List[List[int]]:  # type: ignore
        """
        Find the duplicates of the data.

        Parameters
        ----------
        data: Sequence[str]
            The data to be queried.
        kwargs: Any
            The keyword arguments for the tokenizer function.

        Returns
        -------
        List[List[int]]
            The list of duplicate indices.
        """
        signatures = self.transform(data, **kwargs)

        return self.query(signatures)

    def fit_predict(self, data: Sequence[str], **kwargs) -> List[List[int]]:  # type: ignore
        """
        Fit and predict in one step.

        Parameters
        ----------
        data: Sequence[str]
            The data to be indexed and queried.
        kwargs: Any
            The keyword arguments for the tokenizer function.

        Returns
        -------
        List[List[int]]
            The list of duplicate indices.
        """
        signatures = self.transform(data, **kwargs)
        self.index(signatures)
        return self.query(signatures)

    def transform(self, data: Sequence[str], **kwargs) -> List[Fingerprint]:
        """
        Convert the data into fingerprints.

        Parameters
        ----------
        data: Sequence[str]
            The data to be transformed.
        kwargs: Any
            The keyword arguments for the tokenizer function.

        Returns
        -------
        List[Fingerprint]
            The list of fingerprints.
        """
        with Pool(os.cpu_count()) as pool:
            signatures = pool.map(
                self.__embedder.embed_function(**kwargs),
                data,
                progress_bar=self.verbose,
                progress_bar_options={"desc": "Fingerprinting..."},
            )

        return signatures

    def index(self, signatures: List[Fingerprint]):
        """
        Build a MinHashLSH index for the fingerprints.

        Parameters
        ----------
        signatures: List[Fingerprint]
            The fingerprints to be indexed.
        """
        self.__index = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            storage_config=self.storage_config,
        )
        if self.__index.is_empty() and not self.__index.keys:
            with self.__index.insertion_session() as session:
                for key, minhash in enumerate(signatures):
                    if f"id-{key}" in self.__index.keys:
                        continue
                    session.insert(
                        f"id-{key}",
                        LeanMinHash(seed=self.seed, hashvalues=minhash),
                    )
        return

    def query(self, signatures: List[Fingerprint]) -> List[List[int]]:
        """
        Query the MinHashLSH index for the fingerprints.

        Parameters
        ----------
        signatures: List[Fingerprint]
            The fingerprints to be queried.

        Returns
        -------
        List[List[int]]
            The list of duplicate indices.
        """
        with Pool(os.cpu_count()) as pool:
            neighbors = pool.map(
                lambda signature: [
                    int(x.split("-")[1])
                    for x in self.__index.query(
                        LeanMinHash(seed=self.seed, hashvalues=signature),
                    )
                ],
                signatures,
                progress_bar=self.verbose,
                progress_bar_options={"desc": "Querying..."},
            )

        return neighbors
