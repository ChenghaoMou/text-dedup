# text-dedup

[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/cc66178e49d24908ac1fb2b2dbe4e5b3)](https://www.codacy.com/gh/ChenghaoMou/text-dedup/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/text-dedup&utm_campaign=Badge_Grade)

## Features

-   SOTA embeddings with sentence-transformer
-   Fast de-duplication with annoy
-   Suffix Array and MinHash [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

## Installation

```bash
pip install text-dedup
```

## Usage

-   Using Sentence Transformer

```python
from text_dedup import SentenceTransformerDeduper

df = pd.read_csv('...')

deduper = SentenceTransformerDeduper("distilbert-base-nli-stsb-mean-tokens")
df["group"] = deduper.group(df["text"].values.tolist(), show_progress_bar=True)

# dedup with group indices
df = df.drop_duplicates(["group"], keep="first")
```

-   Using Suffix Array for exact match

```python
from text_dedup import SuffixArray

df = pd.read_csv('...')

deduper = SuffixArray(k=50)
groups, duplicates = deduper.fit_transform(df["text"].values.tolist())

assert len(groups) == len(df), "Invalid number of rows"
assert len(duplicates) == groups.shape[1], "Invalid number of columns"
```

-   Using MinHash for fuzzy match

```python
from text_dedup import MinHashDeduper
deduper = MinHashDeduper(ngram_size=5, threshold=0.3)
groups = deduper.fit_transform(["This is a sentence.", "This is another sentence.", "This is a question.", "hello world"])
assert groups == [0, 0, 2, 3]
```

## Benchmark (w/ a P100)

20k(5%) QQP subset

```text
              precision    recall  f1-score   support

       False       0.75      0.89      0.81     12671
        True       0.73      0.50      0.60      7543

    accuracy                           0.75     20214
   macro avg       0.74      0.70      0.71     20214
weighted avg       0.74      0.75      0.73     20214


--------------------------------------------- benchmark: 1 tests --------------------------------------------
Name (time in s)         Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
-------------------------------------------------------------------------------------------------------------
test_scaling         89.9221  89.9221  89.9221  0.0000  89.9221  0.0000       0;0  0.0111       1          10
-------------------------------------------------------------------------------------------------------------
```
