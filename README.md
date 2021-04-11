![banner](./banner.png)
![PyPI](https://img.shields.io/pypi/v/text-dedup?style=plastic)

Text de-duplication with edit distance, LSH or embeddings. (WIP)

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

## Benchmarks

```
LSH
------------------------------------------------ benchmark: 1 tests ------------------------------------------------
Name (time in ms)          Min       Max      Mean   StdDev    Median      IQR  Outliers     OPS  Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------
test_performance3     767.0355  846.3728  803.1992  31.7007  798.3628  50.2480       2;0  1.2450       5           5
--------------------------------------------------------------------------------------------------------------------

EditDistance
--------------------------------------------- benchmark: 1 tests ---------------------------------------------
Name (time in s)          Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
--------------------------------------------------------------------------------------------------------------
test_performance2     10.7813  11.7912  11.2641  0.3861  11.1549  0.5356       2;0  0.0888       5           5
--------------------------------------------------------------------------------------------------------------

BERT
-------------------------------------------- benchmark: 1 tests -------------------------------------------
Name (time in s)         Min      Max    Mean  StdDev  Median     IQR  Outliers     OPS  Rounds  Iterations
-----------------------------------------------------------------------------------------------------------
test_performance1     8.0105  10.8614  9.4974  1.2967  9.1050  2.3446       3;0  0.1053       5           5
-----------------------------------------------------------------------------------------------------------
```