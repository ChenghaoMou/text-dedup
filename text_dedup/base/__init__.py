from __future__ import annotations

from abc import abstractmethod
from typing import Callable
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

Fingerprint = Union[List[slice], int, np.ndarray, bool]
DuplicationResults = List[List[slice]] | List[int] | List[np.ndarray] | List[bool]


class DuplicateFinder:  # pragma: no cover
    """
    Base class for all duplicate finders. DuplicateFinders offer end-to-end duplicate detection.

    Beware that DuplicateFinder does not remove those duplicates.
    """

    @abstractmethod
    def fit(self, data: Sequence[str]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data: Sequence[str]) -> DuplicationResults:
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
