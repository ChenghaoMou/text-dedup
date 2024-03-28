import os
import pickle  # nosec
import subprocess  # nosec
from collections import defaultdict

import click
import datasets
import pandas as pd
from datasets import Features
from datasets import Sequence
from datasets import Value

from text_dedup.ann_unisim import main as unisim_main
from text_dedup.minhash import main as minhash_main
from text_dedup.simhash import main as simhash_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import UniSimArgs

NUM_PROC = os.cpu_count()


def _recall(row):
    labelled_dups = set(row["duplicates"])
    LEN_LABELLED_DUPLICATES = len(labelled_dups)
    if LEN_LABELLED_DUPLICATES == 0:
        return 1
    dups = set(row["predictions"])
    return len(dups & labelled_dups) / LEN_LABELLED_DUPLICATES


def _precision(row):
    labelled_dups = set(row["duplicates"])
    dups = set(row["predictions"])
    LEN_DUPLICATES = len(dups)
    if LEN_DUPLICATES == 0:
        return 0
    return len(dups & labelled_dups) / LEN_DUPLICATES


def uf2results(path: str, name: str, time: float):
    with open(path, "rb") as f:
        uf = pickle.load(f)  # nosec

    id2cluster = defaultdict(set)
    for idx, cluster in uf.parent.items():
        id2cluster[cluster].add(idx)

    predictions = {
        id2core_id[x["id"]]: {id2core_id[neighbor] for neighbor in id2cluster[uf.find(x["id"])] if neighbor != x["id"]}
        for x in truth
    }
    df = (
        pd.Series(labels)
        .to_frame("duplicates")
        .reset_index()
        .merge(pd.Series(predictions).to_frame("predictions").reset_index(), on="index")
    )
    df["Correct"] = df.apply(lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1).astype(int)
    prediction_summary = {"Correct": df["Correct"].sum(), "Incorrect": df.shape[0] - df["Correct"].sum()}
    prediction_summary["Accuracy"] = round(prediction_summary["Correct"] / df.shape[0], 4)
    recalls = df.apply(_recall, axis=1)
    prediction_summary["Recall"] = round(recalls.mean(), 4)
    precisions = df.apply(_precision, axis=1)
    prediction_summary["Precision"] = round(precisions.mean(), 4)

    df["Class"] = df.apply(classify_in_paper, axis=1)
    df["Class_"] = df.apply(lambda row: inverse(row["Class"]), axis=1)

    f1s = {}
    precisions = {}
    recalls = {}
    for col in ["Class", "Class_"]:
        label_counts = df[col].value_counts()
        precision = label_counts["TP"] / (label_counts["TP"] + label_counts["FP"])
        recall = label_counts["TP"] / (label_counts["TP"] + label_counts["FN"])
        f1 = 2 * precision * recall / (precision + recall)

        precisions[col] = precision
        recalls[col] = recall
        f1s[col] = f1

    table.append(
        [
            name,
            f"{precisions['Class']:.4f}",
            f"{recalls['Class']:.4f}",
            f"{precisions['Class_']:.4f}",
            f"{recalls['Class_']:.4f}",
            f"{(precisions['Class'] + precisions['Class_']) / 2:.4f}",
            f"{df['Correct'].mean():.4f}",
            f"{time:.2f}s",
        ]
    )


def classify_in_paper(record):
    duplicates = set(record["duplicates"])
    predictions = set(record["predictions"])

    LEN_PREDICTIONS = len(predictions)
    LEN_DUPLICATES = len(duplicates)

    # if len(predictions) == 0 it is Negative whether True or not.
    # Hopefully True is more common and short circuit ifs
    if LEN_PREDICTIONS == 0:
        if LEN_DUPLICATES == 0:
            return "TN"
        if LEN_DUPLICATES > 0:
            return "FN"

    # If len(predictions) > 0 it is Positive whether True or not.
    # Hopefully True is more common and short circuit ifs
    # python uses short circuiting so this is more readable and faster
    if LEN_PREDICTIONS > 0:
        if LEN_DUPLICATES > 0 and duplicates.issubset(predictions):
            return "TP"
        if LEN_DUPLICATES == 0 or not duplicates.issubset(predictions):
            return "FP"

    raise ValueError(f"This should not happen {duplicates} {predictions} {len(duplicates)=} {len(predictions)=}")


def inverse(label: str) -> str:
    # inverts the results basically N->P and P->N
    return {"TP": "TN", "FN": "FP", "FP": "FN", "TN": "TP"}[label]


def spark_assignment_to_uf(path: str):
    df = pd.read_parquet(path)
    uf = UnionFind()
    for _, row in df.iterrows():
        uf.union(row["id"], row["component"])

    with open(f"{spark_output}/uf.pkl", "wb") as f:
        pickle.dump(uf, f)
    return uf


def exact_title_results(ds, name: str):
    title2core_ids = defaultdict(set)
    for record in ds:
        title = record["processed_title"]
        core_id = int(record["core_id"])
        title2core_ids[title].add(core_id)

    matches = ds.map(
        lambda row: {"matches": {x for x in title2core_ids[row["processed_title"]] if x != int(row["core_id"])}}
    )
    matches = {int(x["core_id"]): x["matches"] for x in matches}
    ddf = (
        pd.Series(matches)
        .to_frame("predictions")
        .reset_index()
        .merge(pd.Series(labels).to_frame("duplicates").reset_index(), on="index")
    )
    ddf["Correct"] = ddf.apply(lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1).astype(int)
    ddf["Class"] = ddf.apply(lambda row: classify_in_paper(row), axis=1)
    ddf["Class_"] = ddf.apply(lambda row: inverse(row["Class"]), axis=1)

    f1s = {}
    precisions = {}
    recalls = {}
    for col in ["Class", "Class_"]:
        label_counts = ddf[col].value_counts().to_dict()
        precision = label_counts["TP"] / (label_counts["TP"] + label_counts["FP"])
        recall = label_counts["TP"] / (label_counts["TP"] + label_counts["FN"])
        f1 = 2 * precision * recall / (precision + recall)
        precisions[col] = precision
        recalls[col] = recall
        f1s[col] = f1

    table.append(
        [
            name,
            f"{precisions['Class']:.4f}",
            f"{recalls['Class']:.4f}",
            f"{precisions['Class_']:.4f}",
            f"{recalls['Class_']:.4f}",
            f"{(precisions['Class'] + precisions['Class_']) / 2:.4f}",
            f"{ddf['Correct'].mean():.4f}",
            "-",
        ]
    )


if __name__ == "__main__":
    t = Timer()

    table = []

    (
        datasets.load_dataset(
            "pinecone/core-2020-05-10-deduplication", split="train", cache_dir="./cache", num_proc=NUM_PROC
        )
        .map(lambda x: {"text": " ".join((x["processed_title"], x["processed_abstract"])).lower()}, num_proc=NUM_PROC)
        .save_to_disk("temp_inp_ds")
    )

    os.makedirs("temp_inp_paruqet", exist_ok=True)
    (
        datasets.load_dataset(
            "pinecone/core-2020-05-10-deduplication", split="train", cache_dir="./cache", num_proc=NUM_PROC
        )
        .map(
            lambda x, i: {"text": " ".join((x["processed_title"], x["processed_abstract"])).lower(), "id": i},
            num_proc=NUM_PROC,
            with_indices=True,
        )
        .to_pandas()
        .to_parquet("temp_inp_paruqet/data.parquet")
    )

    ds = datasets.load_from_disk("temp_inp_ds")
    truth = ds.map(
        lambda x, idx: {"core_id": x["core_id"], "id": idx, "duplicates": x["labelled_duplicates"]},
        remove_columns=ds.column_names,
        with_indices=True,
        num_proc=NUM_PROC,
        features=Features(
            {
                "core_id": Value("string"),
                "id": Value("int64"),
                "duplicates": Sequence(Value("string")),
            }
        ),
    )
    id2core_id = {x["id"]: int(x["core_id"]) for x in truth}
    labels = {int(x["core_id"]): set(map(int, x["duplicates"])) if x["duplicates"] else set() for x in truth}

    io_args = IOArgs(
        path="./temp_inp_ds",
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output="./temp_output_minhash",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=10000)

    with t("MinHash Spark"):
        spark_output = "./temp_output_spark"
        spark_args = f"""
        spark-submit --executor-memory 86g
            --driver-memory 8g
            --executor-cores 2
            --num-executors 2
            --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12
            --conf spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            text_dedup/minhash_spark.py
            --input ./temp_inp_paruqet
            --output {spark_output}
            --column text
            --index id
            --threshold 0.5
            --num_perm 200
            --b 50
            --r 4
            --ngram 2
            --debug
        """.split("\n")
        subprocess.run(
            [part.strip() for line in spark_args for part in line.strip().split(" ") if part.strip()],
        )  # nosec
        spark_assignment_to_uf(f"{spark_output}-assignment/assignment.parquet")

    with t("SimHash"):
        ctx = click.Context(simhash_main)
        simhash_args = SimHashArgs(bit_diff=7, num_bucket=8, ngram=3)
        io_args.output = simhash_output = "./temp_output_simhash"
        ctx.invoke(
            simhash_main,
            io_args=io_args,
            meta_args=meta_args,
            simhash_args=simhash_args,
        )

    with t("MinHash"):
        ctx = click.Context(minhash_main)
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.5, b=50, r=4)
        io_args.output = minhash_output = "./temp_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    with t("UniSim"):
        ctx = click.Context(unisim_main)
        unisim_args = UniSimArgs(
            store_data=False,
            index_type="approx",
            similarity_threshold=0.86,
        )
        io_args.output = unisim_output = "./temp_output_unisim"
        meta_args.batch_size = 48
        ctx.invoke(
            unisim_main,
            io_args=io_args,
            meta_args=meta_args,
            unisim_args=unisim_args,
        )

    uf2results(f"{unisim_output}/uf.pkl", "UniSim", t.elapsed_times.get("UniSim"))
    uf2results(f"{spark_output}/uf.pkl", "MinHash Spark", t.elapsed_times.get("MinHash Spark"))
    uf2results(f"{minhash_output}/uf.pkl", "MinHash", t.elapsed_times.get("MinHash"))
    uf2results(f"{simhash_output}/uf.pkl", "SimHash", t.elapsed_times.get("SimHash"))
    exact_title_results(ds=ds, name="Exact Title")

    print(
        pd.DataFrame(
            table,
            columns=[
                "Algorithm",
                "Precision (Duplicates)",
                "Recall (Duplicates)",
                "Precision (Non Duplicates)",
                "Recall (Non Duplicates)",
                "Macro F1 score",
                "Accuracy",
                "Time",
            ],
        ).to_markdown(index=False)
    )
