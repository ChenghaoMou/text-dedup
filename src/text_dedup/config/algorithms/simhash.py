# pyright: reportAny=false
# pyright: reportExplicitAny=false
from typing import Literal

from text_dedup.config.algorithms.base import AlgorithmConfig


class SimHashAlgorithmConfig(AlgorithmConfig):
    algo_name: Literal["simhash"] = "simhash"
