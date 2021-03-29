# text-dedup (WIP)
Text deduplication with fuzzy match and more

## Usage

1. Group near duplicates
```python
import pandas as pd
from text_dedup.dedupers import EditDistanceSimilarityDeduper
from text_dedup import group_duplicates

df = pd.read_csv(...)
df_groups = group_duplicates(
    df, 
    deduper=EditDistanceSimilarityDeduper(
        similarity_metric="cosine", 
        threshold=0.8, 
        k=3),
    column="text",
    target_column="__group_label__"
    )

df["__group_label__"].value_counts(dropna=False)
```

2. Remove near duplicates
```python
import pandas as pd
from text_dedup.dedupers import EditDistanceSimilarityDeduper
from text_dedup import drop_duplicates

df = pd.read_csv(...)
df_dedup = drop_duplicates(
    df, 
    deduper=EditDistanceSimilarityDeduper(
        similarity_metric="cosine", 
        threshold=0.8, 
        k=3),
    column="text"
    )

assert df.shape != df_dedup.shape
```

3. Remove semantically similar duplicates
```python
import pandas as pd
from text_dedup.dedupers import PretrainedBERTEmbeddingDeduper
from text_dedup import drop_duplicates

df = pd.read_csv(...)
data_dedup = drop_duplicates(
    df, 
    deduper=PretrainedBERTEmbeddingDeduper(
        model='paraphrase-distilroberta-base-v1',
        threshold=threshold, 
    ),
    column="text"
)
```

## Installation
```bash
pip install text-dedup
```