#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 11:26:04
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from dataclasses import dataclass
from typing import Callable, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from text_dedup.embedders.base import Embedder, Fingerprint


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
    >>> from transformers import AutoTokenizer, AutoModel
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> model = AutoModel.from_pretrained('bert-base-uncased')
    >>> embedder = TransformerEmbedder(tokenizer, model)
    """

    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def embed(self, corpus: List[str], batch_size: int = 8) -> List[Fingerprint]:  # type: ignore
        """
        Embed a list of texts.

        Parameters
        ----------
        corpus : List[str]
            List of texts.
        batch_size : int
            Batch size.

        Returns
        -------
        List[Fingerprint]
            Embeddings.
        """
        embeddings = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i: i + batch_size]
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )

            with torch.no_grad():
                output = self.model(**encodings, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                embeddings.extend(hidden.mean(dim=1).detach().cpu().numpy())

        return embeddings

    def embed_function(self, **kwargs) -> Callable[[str], Fingerprint]:
        raise NotImplementedError("This function is not implemented")
