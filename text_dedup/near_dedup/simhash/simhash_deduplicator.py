import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from mpire import WorkerPool as Pool

from text_dedup.base import Deduplicator
from text_dedup.base import Embedder
from text_dedup.base import Fingerprint
from text_dedup.near_dedup.simhash.simhash_embedder import SimHashEmbedder
from text_dedup.near_dedup.simhash.simhash_index import SimHashIndex
from text_dedup.preprocess import tokenize


@dataclass
class SimHashDeduplicator(Deduplicator):
    """
    Deduplicate text using SimHash.

    Parameters
    ----------
    hamming_distance : int
        The maximum hamming distance between two fingerprints to be considered as duplicates, by default 3.
    num_blocks : int
        The number of blocks to split the fingerprint into, by default 5.
    storage_config : Dict[str, Any]
        The configuration for the storage backend, by default None.
    tokenizer : Callable[[str], List[str]]
        The tokenizer to use, by default tokenize.
    verbose : bool
        Whether to show a progress bar, by default False.
    """
    hamming_distance: int = 3
    num_blocks: int = 5
    storage_config: Optional[Dict[str, Any]] = None
    tokenizer: Callable[..., List[str]] = tokenize
    verbose: bool = False
    __embedder: Embedder = field(init=False, repr=False)
    __index: SimHashIndex = field(init=False, repr=False)

    def __post_init__(self):
        self.__embedder = SimHashEmbedder(tokenizer=self.tokenizer)

    def fit(self, data: Sequence[str], **kwargs) -> 'SimHashDeduplicator':
        """
        Fit the Deduplicator.

        Parameters
        ----------
        data : Sequence[str]
            The text to be processed.
        kwargs: Any
            Additional arguments to be passed to the tokenizer function.

        Returns
        -------
        SimHashDeduplicator
            The fitted Deduplicator.
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

    def fit_predict(self, data: Sequence[str], **kwargs) -> List[List[int]]:
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
        Build a SimHashIndex for the fingerprints.

        Parameters
        ----------
        signatures: List[Fingerprint]
            The fingerprints to be indexed.
        """
        self.__index = SimHashIndex(
            [(i, signature) for i, signature in enumerate(signatures)],
            k=self.hamming_distance,
            b=self.num_blocks,
            storage_config=self.storage_config,
        )

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
                self.__index.get_near_duplicates,
                signatures,
                progress_bar=self.verbose,
                progress_bar_options={"desc": "Querying..."},
            )

        return neighbors
