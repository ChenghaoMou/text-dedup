from typing import List

import pytest
import pandas as pd
from datasets import load_dataset
from text_dedup.dedupers import EditDistanceSimilarityDeduper, PretrainedBERTEmbeddingDeduper, LSHDeduper
from text_dedup import drop_duplicates

@pytest.mark.parametrize(
    ('data', 'threshold', 'expected_size'),  [
        ([
            'Hello world',
            'Hello the world',
            'Hola, yo soy Chenghao'
        ], 0.6, 2)
    ]
)
def test_drop_duplicates_normal(data: List[str], threshold: float, expected_size: int):

    df = pd.DataFrame({"text": data})
    data_dedup = drop_duplicates(
        df, 
        deduper=EditDistanceSimilarityDeduper(
            similarity_metric="cosine", 
            threshold=threshold, 
            k=3),
        column="text"
    )
    
    assert len(data_dedup) == expected_size
    
    data_dedup = drop_duplicates(
        df, 
        deduper=PretrainedBERTEmbeddingDeduper(
            model='paraphrase-distilroberta-base-v1',
            threshold=threshold, 
        ),
        column="text"
    )
    
    assert len(data_dedup) == expected_size

    return

@pytest.mark.parametrize(
    ('data', 'threshold', 'expected_size'),  [
        ([
            'Terresa is loved by his son Jack',
            'Jack loves his mother Terresa',
            'Terresa has a son whose name is Jack and Jack loves his mother very much',
            'This is something very different'
        ], 0.7, 2)
    ]
)
def test_drop_duplicates_semantical(data: List[str], threshold: float, expected_size: int):

    df = pd.DataFrame({"text": data})
    
    data_dedup = drop_duplicates(
        df, 
        deduper=PretrainedBERTEmbeddingDeduper(
            model='paraphrase-distilroberta-base-v1',
            threshold=threshold, 
        ),
        column="text"
    )
    
    assert len(data_dedup) == expected_size

    return

def test_bert(benchmark):

    sample_size: int = 200
    dataset = load_dataset("quora", split="train")
    questions = pd.DataFrame({"text": [r["text"][0] for r in dataset["questions"][:sample_size]] + [r["text"][1] for r in dataset["questions"][:sample_size]]})

    result1 = benchmark.pedantic(
        drop_duplicates,
        kwargs={
        "df": questions, 
        "column": "text", 
        "deduper": PretrainedBERTEmbeddingDeduper(
            model='paraphrase-distilroberta-base-v1',
            threshold=0.7
        )},
        iterations=1,
        rounds=1
    )

    assert len(result1) < sample_size * 2

def test_edit_distance(benchmark):

    sample_size: int = 200
    dataset = load_dataset("quora", split="train")
    questions = pd.DataFrame({"text": [r["text"][0] for r in dataset["questions"][:sample_size]] + [r["text"][1] for r in dataset["questions"][:sample_size]]})

    result2 = benchmark.pedantic(
        drop_duplicates, 
        kwargs={
        "df": questions, 
        "column": "text", 
        "deduper": EditDistanceSimilarityDeduper(
            similarity_metric="cosine", 
            threshold=0.6, 
            k=3
        )},
        iterations=1,
        rounds=1
    )
    assert len(result2) < sample_size * 2

def test_lsh(benchmark):
    
    dataset = load_dataset("quora", split="train")
    sample_size: int = 20000
    questions = pd.DataFrame({"text": [r["text"][0] for r in dataset["questions"][:sample_size]] + [r["text"][1] for r in dataset["questions"][:sample_size]]})

    result2 = benchmark.pedantic(
        drop_duplicates, 
        kwargs={
        "df": questions, 
        "column": "text", 
        "deduper": LSHDeduper(
            threshold=0.5,
        )},
        iterations=1,
        rounds=1
    )
    assert len(result2) < sample_size * 2