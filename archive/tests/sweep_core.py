import os
import pickle  # nosec
from collections import defaultdict
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from os import devnull

import click
import datasets
import pandas as pd
from datasets import Features
from datasets import Sequence
from datasets import Value
from tqdm import tqdm

from text_dedup.simhash import main as simhash_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils.timer import Timer

NUM_PROC = os.cpu_count()


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


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

    return {
        "Algorithm": name,
        "Precision (Duplicates)": precisions["Class"],
        "Recall (Duplicates)": recalls["Class"],
        "Precision (Non Duplicates)": precisions["Class_"],
        "Recall (Non Duplicates)": recalls["Class_"],
        "Macro F1 score": (precisions["Class"] + precisions["Class_"]) / 2,
        "Accuracy": df["Correct"].mean(),
        "Time": time,
    }


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


if __name__ == "__main__":
    t = Timer()

    (
        datasets.load_dataset(
            "pinecone/core-2020-05-10-deduplication", split="train", cache_dir="./cache", num_proc=NUM_PROC
        )
        .map(lambda x: {"text": " ".join((x["processed_title"], x["processed_abstract"])).lower()}, num_proc=NUM_PROC)
        .save_to_disk("temp_inp_ds")
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
        output="",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=10000)
    bit_diff = [7, 6]
    ngram = [3, 4, 5, 6, 7, 8]
    results = []
    for bd in tqdm(bit_diff):
        for ng in tqdm(ngram, leave=False):
            with t("SimHash"), suppress_stdout_stderr():
                ctx = click.Context(simhash_main)
                simhash_args = SimHashArgs(bit_diff=bd, num_bucket=bd + 1, ngram=ng)
                io_args.output = simhash_output = "./temp_output_simhash"
                ctx.invoke(
                    simhash_main,
                    io_args=io_args,
                    meta_args=meta_args,
                    simhash_args=simhash_args,
                )

            metrics = uf2results(f"{simhash_output}/uf.pkl", "SimHash", t.elapsed_times.get("SimHash"))
            metrics["bit_diff"] = bd
            metrics["ngram"] = ng
            results.append(metrics)

            df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
            print(df)

    df.to_csv("./tests/core_simhash_results.tsv", index=False, sep="\t")
