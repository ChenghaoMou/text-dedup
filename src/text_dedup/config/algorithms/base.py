# pyright: reportAny=false
# pyright: reportExplicitAny=false
import os
from typing import Any
from typing import Literal
from typing import override

import numpy as np
from pydantic_settings import BaseSettings


class AlgorithmConfig(BaseSettings):
    algorithm_name: Literal["minhash", "simhash", "bloom_filter", "suffix_array"]
    text_column: str
    index_column: str | None = None
    cluster_column: str = "__CLUSTER__"
    signature_column: str = "__SIGNATURE__"
    seed: int = 42
    num_proc: int = max(1, os.cpu_count() or 1)
    batch_size: int = 1000
    _rng: np.random.RandomState | None
    _internal_index_column: str = "__INDEX__"

    @override
    def model_post_init(self, context: Any) -> None:
        self._rng = np.random.RandomState(self.seed)

    @property
    def internal_index_column(self) -> str:
        return self._internal_index_column
