import os
import pickle  # nosec

import click
import datasets
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from text_dedup.minhash import main as minhash_main
from text_dedup.simhash import main as simhash_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils.preprocessing import news_copy_preprocessing
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind

NUM_PROC = os.cpu_count()


def prepare_data(data_path, label_path, output_path_ds, output_path_spark):
    df = pd.read_json(data_path).T.reset_index()
    labels = pd.read_json(label_path)
    id2data = []
    filename2id = {}
    uf = UnionFind()

    for i, row in df.iterrows():
        id2data.append(
            {
                "filename": str(row["id"]),
                "headline": news_copy_preprocessing(str(row["headline"])),
                "text": news_copy_preprocessing(str(row["headline"] + " " + row["article"])),
                "article": news_copy_preprocessing(str(row["article"])),
                "id": int(i),
            }
        )
        filename2id[id2data[i]["filename"]] = i

    for i, row in labels.iterrows():
        uf.union(filename2id[row[0]], filename2id[row[1]])

    clusters = [None for _ in range(len(df))]
    for i in range(len(df)):
        clusters[i] = uf.find(filename2id[id2data[i]["filename"]])

    ds = datasets.Dataset.from_pandas(pd.DataFrame(id2data))
    ds.save_to_disk(output_path_ds)

    os.makedirs(output_path_spark, exist_ok=True)
    pd.DataFrame(id2data).to_parquet(output_path_spark + "/data.parquet")

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

    test_data = ("./data/test_inf_data.json", "./data/full_test_gt.json")
    val_data = ("./data/1955_inf_data.json", "./data/1955_gt.json")
    labels = prepare_data(*test_data, output_path_ds, output_path_spark)

    io_args = IOArgs(
        path=output_path_ds,
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output="./news_output_minhash",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="article", batch_size=10000)

    # TODO: hyperparameter tuning
    with t("MinHash"):
        ctx = click.Context(minhash_main)
        minhash_args = MinHashArgs(num_perm=256, ngram=2, min_length=0, threshold=0.45)
        io_args.output = minhash_output = "./news_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    # TODO: hyperparameter tuning
    with t("SimHash"):
        ctx = click.Context(simhash_main)
        simhash_args = SimHashArgs(bit_diff=12, num_bucket=13, ngram=5)
        io_args.output = simhash_output = "./temp_output_simhash"
        ctx.invoke(
            simhash_main,
            io_args=io_args,
            meta_args=meta_args,
            simhash_args=simhash_args,
        )

    print(f"MinHash ARI: {uf2results(labels, f'{minhash_output}/uf.pkl')}")
    print(f"SimHash ARI: {uf2results(labels, f'{simhash_output}/uf.pkl')}")
