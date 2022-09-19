#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 11:26:04
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

# TODO Update to SentenceTransformers

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import List
from typing import Optional

from sentence_transformers import SentenceTransformer

from text_dedup.base import Embedder
from text_dedup.base import Fingerprint


@dataclass
class TransformerEmbedder(Embedder):
    """
    Transformer-based embedder.

    Parameters
    ----------
    tokenizer: PreTrainedTokenizer
        Tokenizer to use.
    model: PreTrainedModel
        Model to use.

    Examples
    --------
    >>> model = TransformerEmbedder("bert-base-uncased")
    >>> len(model.embed(["Hello world!"]))
    1
    >>> type(model.embed(["Hello world!"]))
    <class 'list'>
    >>> type(model.embed(["Hello world!"])[0])
    <class 'numpy.ndarray'>
    >>> model.embed(["Hello world!"])[0].shape
    (768,)
    """

    model_name: str
    __model: SentenceTransformer = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.__model = SentenceTransformer(self.model_name)

    def embed(self, corpus: List[str], **kwargs) -> List[Fingerprint]:  # type: ignore
        """
        Embed a list of texts.

        Parameters
        ----------
        corpus : List[str]
            List of texts.
        kwargs: dict
            Keyword arguments to pass to the model's encode function.

        Returns
        -------
        List[Fingerprint]
            Embeddings.
        """
        return [embedding for embedding in self.__model.encode(corpus, **kwargs)]

    def embed_function(self, **kwargs) -> Callable[[str], Fingerprint]:

        return lambda x: self.embed([x], **kwargs)[0]
