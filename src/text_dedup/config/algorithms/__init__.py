from .base import AlgorithmConfig
from .bloom import BloomFilterAlgorithmConfig
from .minhash import MinHashAlgorithmConfig
from .simhash import SimHashAlgorithmConfig
from .suffix_array import SuffixArrayAlgorithmConfig

type AlgoConfig = (
    MinHashAlgorithmConfig | SimHashAlgorithmConfig | BloomFilterAlgorithmConfig | SuffixArrayAlgorithmConfig
)

__all__ = [
    "AlgoConfig",
    "AlgorithmConfig",
    "BloomFilterAlgorithmConfig",
    "MinHashAlgorithmConfig",
    "SimHashAlgorithmConfig",
    "SuffixArrayAlgorithmConfig",
]
