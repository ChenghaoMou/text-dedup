from text_dedup.exact_dedup.suffix_array import GoogleSuffixArrayDeduplicator
from text_dedup.exact_dedup.suffix_array import PythonSuffixArrayDeduplicator
from text_dedup.exact_dedup.suffix_array.base import SuffixArrayDeduplicator

__all__ = ['PythonSuffixArrayDeduplicator',
           'GoogleSuffixArrayDeduplicator', 'SuffixArrayDeduplicator']
