# pyright: reportAny=false
# pyright: reportExplicitAny=false
from typing import Literal

from text_dedup.config.algorithms.base import AlgorithmConfig


class SuffixArrayAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["suffix_array"] = "suffix_array"
