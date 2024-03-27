import os
import pickle  # nosec
import shutil
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from copy import deepcopy
from os import devnull

import click
import datasets
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from text_dedup.simhash import main as simhash_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils.preprocess import news_copy_preprocessing
from text_dedup.utils.timer import Timer

NUM_PROC = os.cpu_count()


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def prepare_data(dataset, output_path_ds, output_path_spark):
    clusters = dataset["cluster"]
    dataset = dataset.map(
        lambda x: {"text": news_copy_preprocessing(x["article"])},
        num_proc=NUM_PROC,
    )
    dataset.save_to_disk(output_path_ds)
    os.makedirs(output_path_spark, exist_ok=True)
    dataset.to_pandas().to_parquet(output_path_spark + "/data.parquet")

    return clusters


def uf2results(labels, output_path):
    with open(output_path, "rb") as f:
        uf = pickle.load(f)  # nosec

    predictions = [uf.find(i) for i in range(len(labels))]
    return adjusted_rand_score(labels, predictions)


if __name__ == "__main__":
    t = Timer()

    output_path_ds = "news_input_ds"
    output_path_spark = "news_input_spark"

    test_data = datasets.load_dataset("chenghao/NEWS-COPY-eval", split="val")
    labels = prepare_data(test_data, output_path_ds, output_path_spark)

    io_args = IOArgs(
        path=output_path_ds,
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output="",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=10000, idx_column="idx")

    bit_diff = [10, 11, 12, 13, 14, 15]
    ngram = [3, 4, 5, 6, 7, 8, 9, 10]
    results = []
    ctx = click.Context(simhash_main)
    io_args.output = simhash_output = "./temp_output_simhash"
    for bd in tqdm(bit_diff):
        for ng in tqdm(ngram, leave=False):
            if os.path.exists(simhash_output):
                shutil.rmtree(simhash_output)
            if os.path.exists(".cache"):
                shutil.rmtree(".cache")
            with t("SimHash"), suppress_stdout_stderr():
                ctx.invoke(
                    simhash_main,
                    io_args=io_args,
                    meta_args=meta_args,
                    simhash_args=SimHashArgs(bit_diff=bd, num_bucket=bd + 1, ngram=ng),
                )

            metrics = {"ARI": uf2results(labels, f"{simhash_output}/uf.pkl")}
            metrics["time"] = t.elapsed_times.get("SimHash")
            metrics["bit_diff"] = bd
            metrics["ngram"] = ng
            results.append(deepcopy(metrics))

            df = pd.DataFrame(results).sort_values("ARI", ascending=False)
            print(df)
    df.to_csv("./tests/news_simhash_results.tsv", index=False, sep="\t")
