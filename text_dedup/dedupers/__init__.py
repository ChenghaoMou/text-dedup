#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-13 09:20:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import List, Union, Optional

import numpy as np
import dask.bag as bag
import pandas as pd
from nltk import ngrams
from numpy import linalg as LA
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer, util

class Deduper:
    
    @staticmethod
    def extract(corpus: Union[pd.DataFrame, pd.Series, List[str]], column: Optional[str] = None) -> List[str]:

        if isinstance(corpus, list):
            return corpus
        elif isinstance(corpus, pd.Series):
            return corpus.values.tolist()
        elif isinstance(corpus, pd.DataFrame) and column is not None:
            return corpus[column].values.tolist()
        else:
            raise ValueError('Invalid input, please check your data')
    
    def group(self, corpus: Union[pd.DataFrame, pd.Series, List[str]], column: Optional[str] = None) -> List[List[bool]]:
        """Group duplicates and return a boolean matrix indicating whether two docs are similar.

        Parameters
        ----------
        corpus : Union[pd.DataFrame, pd.Series, List[str]]
            Generic collection of documents
        column : Optional[str], optional
            Target column of a dataframe, by default None

        Returns
        -------
        List[List[bool]]
            A boolean matrix
        """

        raise NotImplementedError('This function is not implemented')

class LSHDeduper(Deduper):

    def __init__(self, threshold: float = 0.5, num_perm: int = 128, shingle_size: int=3):

        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
    
    def group(self, corpus: Union[pd.DataFrame, pd.Series, List[str]], column: Optional[str] = None) -> List[List[bool]]:
        
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        corpus = self.extract(corpus, column=column)

        def hashing(idx, shingles, num_perm):
            min_hash = MinHash(num_perm)
            for shingle in shingles:
                min_hash.update(''.join(shingle).encode('utf-8'))
            return (idx, min_hash)

        # Only take parallel processing when the data size is larger than 2k
        if len(corpus) <= 2_000:
            hashes = []
            for idx, doc in enumerate(corpus):
                _, min_hash = hashing(idx, ngrams(doc, self.shingle_size), self.num_perm)
                lsh.insert(f'm{idx}', min_hash)
                hashes.append((idx, min_hash))
        else:
            hashes = bag.from_sequence([(i, list(ngrams(doc, self.shingle_size)), self.num_perm) for i, doc in enumerate(corpus)])
            hashes = hashes.map(lambda x: hashing(*x)).compute()
            for idx, min_hash in hashes:
                lsh.insert(f'm{idx}', min_hash)

        matrix = np.zeros((len(corpus), len(corpus)))

        for idx, min_hash in hashes:
            candidates = lsh.query(min_hash)
            for candidate in candidates:
                matrix[idx][int(candidate[1:])] = 1.0
        
        return matrix.astype(bool)

class EditDistanceSimilarityDeduper(Deduper):

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
        >>> deduper = EditDistanceSimilarityDeduper('cosine', threshold=0.7, k=2)
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
    
    def group(self, corpus: Union[pd.DataFrame, pd.Series, List[str]], column: Optional[str] = None) -> List[List[bool]]:
        
        corpus = self.extract(corpus, column=column)

        def calculate(idx, doc, corpus):
            return (idx, [self.similarity_metric.similarity(doc, other) for other in corpus])
        
        matrix = np.zeros((len(corpus), len(corpus)))

        # Only take parallel processing when the data size is larger than 2k
        if len(corpus) <= 2_000:
            for idx, doc in enumerate(corpus):
                _, similarities = calculate(idx, doc, corpus)
                for j, similarity in enumerate(similarities):
                    matrix[idx][j] = int(similarity >= self.threshold)
        else:
            docs = bag.from_sequence(list(enumerate(corpus)))
            docs = docs.map(lambda x: calculate(x[0], x[1], corpus)).compute()
            for idx, similarities in docs:
                for j, similarity in similarities:
                    matrix[idx][j] = int(similarity >= self.threshold)
        
        return matrix

class PretrainedBERTEmbeddingDeduper(Deduper):

    def __init__(self, model: str, threshold: float):
        self.model = SentenceTransformer(model)
        self.threshold = threshold
    
    def group(self, corpus: Union[pd.DataFrame, pd.Series, List[str]], column: Optional[str] = None) -> List[List[bool]]:
        corpus = self.extract(corpus, column=column)
        embeddings = self.model.encode(corpus, convert_to_tensor=True)
        return (util.pytorch_cos_sim(embeddings, embeddings) >= self.threshold).numpy().tolist()