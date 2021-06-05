# text-dedup

## Installation

```bash
pip install text-dedup
```

## Usage

```python
from text_dedup import SentenceTransformerDeduper

df = pd.read_csv('...')

deduper = SentenceTransformerDeduper("distilbert-base-nli-stsb-mean-tokens")
df["group"] = deduper.group(df["text"].values.tolist(), show_progress_bar=True)

# dedup with group indices
df = df.drop_duplicates(["group"], keep="first")
```

## Benchmark (w/ GPU)

20k QQP subset

```

--------------------------------------------- benchmark: 1 tests --------------------------------------------
Name (time in s)         Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
-------------------------------------------------------------------------------------------------------------
test_scaling         89.9221  89.9221  89.9221  0.0000  89.9221  0.0000       0;0  0.0111       1          10
-------------------------------------------------------------------------------------------------------------
```
