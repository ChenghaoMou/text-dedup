![banner](./banner.png)
[![PyPI version](https://badge.fury.io/py/text-dedup.svg)](https://badge.fury.io/py/text-dedup) ![Coverage](./coverage.svg)


Text de-duplication with edit distance, LSH or embeddings. (WIP)

## Usage

1. Group near duplicates with `EditDistanceSimilarityDeduper` or `LSHDeduper`
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

3. Remove semantically similar duplicates using `PretrainedBERTEmbeddingDeduper`
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

## Benchmarks

- 400 samples
```
------------------------------------------------------------------------------------------- benchmark: 3 tests ------------------------------------------------------------------------------------------
Name (time in ms)              Min                    Max                   Mean              StdDev                 Median                 IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_lsh                  748.3384 (1.0)         752.1715 (1.0)         750.5136 (1.0)        1.7357 (1.0)         751.4906 (1.0)        2.9455 (1.0)           1;0  1.3324 (1.0)           5           5
test_bert               7,233.6232 (9.67)      8,480.0729 (11.27)     8,058.8376 (10.74)    513.7158 (295.97)    8,311.9608 (11.06)    681.7020 (231.44)        1;0  0.1241 (0.09)          5           5
test_edit_distance     10,040.8134 (13.42)    10,290.2110 (13.68)    10,165.0379 (13.54)    113.2858 (65.27)    10,111.8537 (13.46)    196.6889 (66.78)         3;0  0.0984 (0.07)          5           5
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

- 40000 samples (`PretrainedBERTEmbeddingDeduper` and `EditDistanceSimilarityDeduper` might not be scaling well to large datasets)
```
----------------------------------------------- benchmark: 1 tests ----------------------------------------------
Name (time in s)          Min       Max      Mean  StdDev    Median     IQR  Outliers     OPS  Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------
test_lsh             714.5114  714.5114  714.5114  0.0000  714.5114  0.0000       0;0  0.0014       1           1
-----------------------------------------------------------------------------------------------------------------
```