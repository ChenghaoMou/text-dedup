#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-13 09:20:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Dict, List

import numpy as np
import torch
from numpy import linalg as LA
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from sentence_transformers import SentenceTransformer, util

class Deduper:
    
    def compare(self, this: str, other: str) -> bool:
        return this == other
    
    def batch_compare(self, this: List[str]) -> List[List[bool]]:
        return [[self.compare(x, y) for y in this] for x in this]

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
    
    def compare(self, this: str, other: str) -> bool:
        
        if self.similarity_metric is None:
            raise ValueError("Unknown similarity_metric")

        return self.similarity_metric.similarity(this, other) >= self.threshold

class PretrainedWordEmbeddingDeduper(Deduper):

    def __init__(self, embedding_matrix: Dict[str, np.ndarray], threshold: float):
        """Traditional word-embedding-based similarity deduper. Current implementation takes one pair at a time
        which makes the overall time complexity O(n**2). Suitable for small datasets.

        Parameters
        ----------
        embedding_matrix : Dict[str, np.ndarray]
            A dictionary mapping from a word to its embedding
        threshold : float
            Similarity threshold, anything larger than which will be considered
            as a duplicate

        Examples
        --------
        >>> deduper = PretrainedWordEmbeddingDeduper({'hello': [0, 1], 'world': [1, 0], 'english': [1, 0]}, threshold=0.5)
        >>> deduper.compare('hello world', 'hello english')
        True
        """
        self.embedding_matrix = embedding_matrix
        self.embedding_size = np.asarray(embedding_matrix[list(embedding_matrix.keys())[0]]).reshape(1, -1).shape[-1]
        self.threshold = threshold
    
    def compare(self, this: str, other: str) -> bool:
        x, y = self.embed(this), self.embed(other)
        return x.dot(y.T).item() / (LA.norm(x) * LA.norm(y)) >= self.threshold
    
    def embed(self, text: str) -> np.ndarray:
        return np.sum(
            [np.asarray(self.embedding_matrix.get(w, np.zeros(self.embedding_size))).reshape(1, -1) for w in text.split(' ')], axis=0
        )

    def batch_compare(self, this: List[str]) -> List[List[bool]]:
        embeddings1 = np.asarray([self.embed(t) for t in this]) # [N, H]
        return (util.pytorch_cos_sim(torch.from_numpy(embeddings1), torch.from_numpy(embeddings1)) >= self.threshold).numpy()

class PretrainedBERTEmbeddingDeduper(Deduper):

    def __init__(self, model: str, threshold: float):
        self.model = SentenceTransformer(model)
        self.threshold = threshold
    
    def compare(self, this: str, other: str) -> bool:
        embeddings1 = self.model.encode([this], convert_to_tensor=True)
        embeddings2 = self.model.encode([other], convert_to_tensor=True)

        #Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.item() >= self.threshold

    def batch_compare(self, this: List[str]) -> List[List[bool]]:
        embeddings1 = self.model.encode(this, convert_to_tensor=True)

        return (util.pytorch_cos_sim(embeddings1, embeddings1) >= self.threshold).numpy()