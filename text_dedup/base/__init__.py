from __future__ import annotations

from abc import abstractmethod
from typing import Callable
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

Fingerprint = Union[int, List[slice], np.ndarray]


class Deduplicator:  # pragma: no cover
    """
    Base class for all deduplicators. Deduplicators offers end-to-end duplicate detection.

    This is designed for datasets that comfortably fit into memory. For larger datasets, use Embedder instead.
    """

    @abstractmethod
    def fit(self, data: Sequence[str]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data: Sequence[str]) -> List[List[slice]] | List[Fingerprint]:
        raise NotImplementedError()

    @abstractmethod
    def fit_predict(self, data: Sequence[str]):
        raise NotImplementedError()


class Embedder:  # pragma: no cover
    """
    Base class for all embedders. Embedders are used to transform a string into a fingerprint.

    This is often used together with a clustering method to group similar fingerprints.
    """
    @abstractmethod
    def embed(self, corpus: Sequence[str], **kwargs) -> List[Fingerprint]:
        raise NotImplementedError

    @abstractmethod
    def embed_function(self, **kwargs) -> Callable[[str], Fingerprint]:
        raise NotImplementedError
