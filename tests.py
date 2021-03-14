import pytest
import pandas as pd
from text_dedup.dedupers import EditDistanceSimilarityDeduper
from text_dedup import drop_duplicates

@pytest.mark.parametrize(
    'path', [
        "/Users/chenghaomou/Downloads/quora-question-pairs/test.csv",
    ]
)
def test_drop_duplicates(path: str):

    df = pd.read_csv("/Users/chenghaomou/Downloads/quora-question-pairs/test.csv")
    df = df.sample(frac=0.0001)
    df = df.dropna()
    data = pd.DataFrame(df["question1"].values.tolist() + df["question2"].values.tolist(), columns=["text"])
    data_dedup = drop_duplicates(
        data, 
        deduper=EditDistanceSimilarityDeduper(
            similarity_metric="cosine", 
            threshold=0.8, 
            k=3),
        column="text"
        )
    
    print(data.shape, data_dedup.shape)

    return