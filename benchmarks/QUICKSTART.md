# Benchmark Quickstart Guide

## Installation

Ensure you have the benchmark dependencies:

```bash
# Install scikit-learn for ARI metric
uv add scikit-learn
```

## Running Benchmarks

### Option 1: Use Just (Recommended)

```bash
# Run everything
just benchmark-all

# Or run specific benchmarks
just benchmark-core           # CORE dataset (MinHash + SimHash)
just benchmark-news           # NEWS-COPY dataset (MinHash + SimHash)
just benchmark-core-minhash   # CORE with MinHash only
just benchmark-news-simhash   # NEWS-COPY with SimHash only
```

### Option 2: Direct Python Execution

```bash
# Run all benchmarks
python -m benchmarks.run_benchmark --dataset all --algorithms all

# Run specific combinations
python -m benchmarks.run_benchmark --dataset core --algorithms minhash
python -m benchmarks.run_benchmark --dataset news --algorithms minhash,simhash
```

## What Gets Benchmarked?

### CORE Dataset

- **Dataset**: `pinecone/core-2020-05-10-deduplication`
- **Task**: Academic paper deduplication
- **Metrics**: Precision, Recall, F1, Accuracy
- **Algorithms**: MinHash, SimHash

### NEWS-COPY Dataset

- **Dataset**: `chenghao/NEWS-COPY-eval`
- **Task**: News article near-duplicate clustering
- **Metrics**: Adjusted Rand Index (ARI)
- **Algorithms**: MinHash, SimHash

## Expected Output

```
================================================================================
TEXT DEDUPLICATION BENCHMARK SUITE
================================================================================

Datasets: core, news
Algorithms: minhash, simhash

====================================================================================================
Loading CORE dataset (pinecone/core-2020-05-10-deduplication)...
====================================================================================================
Dataset loaded: XXXX documents

--------------------------------------------------------------------------------
Running MinHash benchmark on CORE dataset...
--------------------------------------------------------------------------------
MinHash completed in XX.XXs

====================================================================================================
Benchmark Results on pinecone/core-2020-05-10-deduplication
====================================================================================================
Algorithm            | P(Dup)   | R(Dup)   | P(Non)   | R(Non)   | Macro F1 | Accuracy | Time
----------------------------------------------------------------------------------------------------
MinHash              | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XX.XXs
SimHash              | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XX.XXs
====================================================================================================

[Similar output for NEWS-COPY dataset]

================================================================================
BENCHMARKS COMPLETED SUCCESSFULLY
================================================================================
```

## Customizing Benchmarks

To modify benchmark parameters:

1. Edit config files in `configs/`:

   - `benchmark_core_minhash.toml`
   - `benchmark_core_simhash.toml`
   - `benchmark_news_minhash.toml`
   - `benchmark_news_simhash.toml`

2. Adjust hyperparameters like:

   - `num_perm` (MinHash permutations)
   - `threshold` (similarity threshold)
   - `ngram_size` (token n-grams)
   - `bit_diff` (SimHash bit difference)

3. Re-run benchmarks with `just benchmark-all`

## Troubleshooting

### Issue: Config file not found

**Solution**: Ensure you're running from the project root directory where `configs/` exists.

### Issue: Dataset download fails

**Solution**: Check your internet connection. HuggingFace datasets will be cached after first download.

### Issue: Out of memory

**Solution**: Reduce `num_proc` in config files or use a machine with more RAM.

### Issue: Results differ from README

**Solution**: Hyperparameters may differ. Check config files match the paper/README settings.

## Adding New Algorithms

To benchmark a new algorithm:

1. Create `run_<algorithm>_benchmark()` function in `benchmark_core.py` or `benchmark_news.py`
2. Add config file in `configs/benchmark_<dataset>_<algorithm>.toml`
3. Update `run_benchmark.py` to include the new algorithm
4. Add just command in `justfile`

Example:

```python
# In benchmark_core.py
def run_bloom_benchmark(config: Config, labels: dict, id_to_core_id: dict) -> tuple[dict, float]:
    timer = Timer()
    with timer("BloomFilter"):
        bloom_main(config)
    # ... evaluation logic
    return metrics, timer.timings["BloomFilter"]
```

## Performance Tips

- First run will be slower (downloads datasets)
- Subsequent runs use cached data
- Use `clean_cache = false` in configs to reuse intermediate results
- Run single-algorithm benchmarks during development
- Run `benchmark-all` for final validation
