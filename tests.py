from typing import List
import pytest
import pandas as pd
from text_dedup.dedupers import EditDistanceSimilarityDeduper, PretrainedBERTEmbeddingDeduper
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