#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-13 09:20:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import abc
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from nltk import ngrams
from numpy import linalg as LA
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer, util
from sklearn.base import TransformerMixin
from alive_progress import alive_bar

class Grouper(TransformerMixin):

    @abc.abstractmethod
    def transform(self, X) -> List[int]:
        raise NotImplementedError('This function is not implemented')

    def fit_transform(self, X) -> List[int]:
        return self.transform(X)

    @staticmethod
    def extract(corpus: Union[pd.Series, List[str]]) -> List[str]:
        """Extract text from a pd.Series or a list of strings

        Parameters
        ----------
        corpus : Union[pd.Series, List[str]]
            Input data

        Returns
        -------
        List[str]
            List of extracted text

        Raises
        ------
        ValueError
            Invalid input
        """
        if isinstance(corpus, list):
            return corpus
        elif isinstance(corpus, pd.Series):
            return corpus.values.tolist()
        else:
            raise ValueError(f'Invalid input, please check your parameter corpus: {type(corpus)}')

    @staticmethod
    def matrix2groups(matrix: np.ndarray) -> List[int]:
        """Convert a similarity matrix into a list of group indices.

        Parameters
        ----------
        matrix : np.ndarray
            Similarity matrix

        Returns
        -------
        List[int]
            List of group indices
        """
        duplicates: List[List[bool]] = []
        h = len(matrix)
        for i in range(h):
            duplicates.append([duplicates[j][i] if j < i else matrix[i][j] if i != j else True for j in range(h)])
        
        parent = {}

        def find_parent(i):
            if parent.get(i, i) == i:
                return i
            
            return find_parent(parent[i])

        for i in range(h):
            parent[i] = find_parent(i)
            for j in range(h):
                if j >= i: continue
                if bool(duplicates[i][j]) is True:
                    parent[i] = min(find_parent(i), find_parent(j))
        
        return [parent[i] for i in range(h)]

class LSHGrouper(Grouper):

    def __init__(self, threshold: float = 0.5, num_perm: int = 128, shingle_size: int=3):
        
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
    
    def transform(self, X: Union[pd.Series, List[str]]) -> List[int]:
        
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        
        corpus = self.extract(X)
        hashes = []

        def hashing(idx, shingles, num_perm):
            min_hash = MinHash(num_perm)
            for shingle in shingles:
                min_hash.update(''.join(shingle).encode('utf-8'))
            return (idx, min_hash)

        with alive_bar(len(corpus)) as bar:
            for idx, doc in enumerate(corpus):
                _, min_hash = hashing(idx, ngrams(doc, self.shingle_size), self.num_perm)
                lsh.insert(f'm{idx}', min_hash)
                hashes.append((idx, min_hash))
                bar()

        matrix = np.zeros((len(corpus), len(corpus)))

        for idx, min_hash in hashes:
            candidates = lsh.query(min_hash)
            for candidate in candidates:
                matrix[idx][int(candidate[1:])] = 1.0
        
        return self.matrix2groups(matrix)

class EditDistanceSimilarityGrouper(Grouper):

    def __init__(self, similarity_metric: str, threshold: float, **kwargs):
        """Edit-Distance-based similarity deduper. Current implementation takes one pair at a time
        which makes the overall time complexity O(n**2). Suitable for small datasets.

        Parameters
        ----------
        similarity_metric : str
            Name of the distance metric
        threshold : float
            The similarity threshold, anything larger than which will be considered
            as a duplicate
        **kargs : type
            Any additional keyword arguments passed to the distance metric, see strsimpy for details

        Examples
        --------
        >>> deduper = EditDistanceSimilarityGrouper('cosine', threshold=0.7, k=2)
        >>> deduper.compare('this is a message', 'this is another message')
        True
        >>> deduper.compare('this is a message', 'hello world')
        False
        """
        self.similarity_metric = {
            "cosine": Cosine(**kwargs),
            "jaccard": Jaccard(**kwargs),
            "jaro_winkler": JaroWinkler(threshold=threshold),
            "normalized_levenshtein": NormalizedLevenshtein
        }.get(similarity_metric, None)
        self.threshold = threshold
    
    def transform(self, X: Union[pd.Series, List[str]]) -> List[int]:
        
        corpus = self.extract(X)

        def calculate(idx, doc, corpus):
            return (idx, [self.similarity_metric.similarity(doc, other) for other in corpus])
        
        matrix = np.zeros((len(corpus), len(corpus)))

        with alive_bar(len(corpus)) as bar:
            for idx, doc in enumerate(corpus):
                _, similarities = calculate(idx, doc, corpus)
                for j, similarity in enumerate(similarities):
                    matrix[idx][j] = int(similarity >= self.threshold)
                bar()
        
        return self.matrix2groups(matrix)

class PretrainedBERTEmbeddingGrouper(Grouper):

    def __init__(self, model: str, threshold: float):
        self.model = SentenceTransformer(model)
        self.threshold = threshold
    
    def transform(self, X: Union[pd.Series, List[str]]) -> List[int]:
        corpus = self.extract(X)
        embeddings = self.model.encode(corpus, convert_to_tensor=True)
        
        matrix = (util.pytorch_cos_sim(embeddings, embeddings) >= self.threshold).numpy().tolist()
        return self.matrix2groups(matrix)