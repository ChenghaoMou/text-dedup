from typing import TypeAlias

from .base import AlgorithmConfig
from .bloom import BloomFilterAlgorithmConfig
from .minhash import MinHashAlgorithmConfig
from .simhash import SimHashAlgorithmConfig
from .suffix_array import SuffixArrayAlgorithmConfig

AlgoConfig: TypeAlias = MinHashAlgorithmConfig | SimHashAlgorithmConfig | BloomFilterAlgorithmConfig

__all__ = [
    "AlgoConfig",
    "AlgorithmConfig",
    "BloomFilterAlgorithmConfig",
    "MinHashAlgorithmConfig",
    "SimHashAlgorithmConfig",
    "SuffixArrayAlgorithmConfig",
]
