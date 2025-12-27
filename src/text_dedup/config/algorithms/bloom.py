# pyright: reportAny=false
# pyright: reportExplicitAny=false
from typing import Literal

from rbloom import Bloom

from text_dedup.config.algorithms.base import AlgorithmConfig


class BloomFilterAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["bloomfilter"] = "bloomfilter"
    max_elements: int
    error_rate: float

    def get_filter(self) -> Bloom:
        return Bloom(self.max_elements, self.error_rate)
