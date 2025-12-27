# Benchmarks

This directory contains benchmark scripts for evaluating deduplication algorithms on standard datasets.

## Datasets

### 1. CORE Dataset (`pinecone/core-2020-05-10-deduplication`)

- **Purpose**: Academic paper deduplication
- **Metric**: Precision, Recall, F1, Accuracy
- **Ground Truth**: Labeled duplicate pairs
- **Script**: `benchmark_core.py`

### 2. NEWS-COPY Dataset (`chenghao/NEWS-COPY-eval`)

- **Purpose**: News article near-duplicate detection
- **Metric**: Adjusted Rand Index (ARI)
- **Ground Truth**: Cluster labels
- **Script**: `benchmark_news.py`

## Usage

### Quick Start with Just

The easiest way to run benchmarks is using the `just` command:

```bash
# Run all benchmarks (both datasets, all algorithms)
just benchmark-all

# Run only CORE dataset benchmarks
just benchmark-core

# Run only NEWS-COPY dataset benchmarks
just benchmark-news

# Run specific algorithm on specific dataset
just benchmark-core-minhash
just benchmark-core-simhash
just benchmark-news-minhash
just benchmark-news-simhash
```

### Manual Execution

You can also run the benchmark script directly as a module:

```bash
# All datasets, all algorithms
python -m benchmarks.run_benchmark --dataset all --algorithms all

# Specific dataset
python -m benchmarks.run_benchmark --dataset core --algorithms minhash,simhash
python -m benchmarks.run_benchmark --dataset news --algorithms minhash

# Multiple datasets
python -m benchmarks.run_benchmark --dataset core news --algorithms all
```

### Configuration Files

Configuration files are located in `configs/`:

- `benchmark_core_minhash.toml` - MinHash on CORE dataset
- `benchmark_core_simhash.toml` - SimHash on CORE dataset
- `benchmark_news_minhash.toml` - MinHash on NEWS-COPY dataset
- `benchmark_news_simhash.toml` - SimHash on NEWS-COPY dataset

### Customizing Benchmarks

1. **Modify hyperparameters** in the config TOML files
2. **Add new algorithms** by creating corresponding `run_*_benchmark` functions
3. **Add new datasets** by creating new benchmark scripts following the existing patterns

## Evaluation Metrics

### CORE Dataset Metrics

- **Precision (Duplicates)**: Of all predicted duplicates, how many are true duplicates?
- **Recall (Duplicates)**: Of all true duplicates, how many were found?
- **Precision (Non-Duplicates)**: Of all predicted non-duplicates, how many are true non-duplicates?
- **Recall (Non-Duplicates)**: Of all true non-duplicates, how many were found?
- **Macro F1**: Harmonic mean of precision and recall
- **Accuracy**: Percentage of exact matches between predictions and ground truth

### NEWS-COPY Dataset Metrics

- **ARI (Adjusted Rand Index)**: Measures similarity between predicted and true clustering
  - Range: [-1, 1], where 1 is perfect clustering
  - Adjusts for random chance

## Reproducing README Results

To reproduce the benchmark results mentioned in the main README:

1. Run all benchmark scripts with the provided configs
2. Compare results with the tables in the README
3. Adjust hyperparameters if needed (see config files)

### Expected Results (from README)

**CORE Dataset:**

- MinHash: Accuracy ~0.924, Macro F1 ~0.953
- SimHash: Accuracy ~0.832, Macro F1 ~0.848

**NEWS-COPY Dataset:**

- MinHash: ARI ~0.742
- SimHash: ARI ~0.612

## Dependencies

```bash
pip install scikit-learn pandas
```

All other dependencies are already in the main project requirements.
