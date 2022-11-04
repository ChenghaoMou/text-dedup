from text_dedup.exact_dedup.exact_hash import ExactHash
from text_dedup.exact_dedup.exact_hash import \
    ExactHash as ThePileExactDeduplicator
from text_dedup.exact_dedup.suffix_array import GoogleSuffixArray
from text_dedup.exact_dedup.suffix_array import PythonSuffixArray
from text_dedup.exact_dedup.suffix_array.base import SuffixArray

__all__ = [
    'PythonSuffixArray',
    'GoogleSuffixArray',
    'SuffixArray',
    'ExactHash',
    'ThePileExactDeduplicator',
]
