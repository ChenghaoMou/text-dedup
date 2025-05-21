import json
from collections import defaultdict
from zipfile import ZipFile

import datasets
import pandas as pd

from text_dedup.utils.union_find import UnionFind

# https://www.dropbox.com/sh/so3iw4xecayyrow/AAAiy5FhDf0WpUeHFzxO1SIza?dl=0
# https://github.com/dell-research-harvard/NEWS-COPY


def preprocess_split(file, data, gt):
    with ZipFile(file, "r") as zip:
        pairs = json.load(zip.open(gt))
        data = json.load(zip.open(data))

    records = []
    image_id_to_idx = {}
    for i, (key, values) in enumerate(data.items()):
        records.append(values | {"idx": i})
        image_id_to_idx[key] = i

    uf = UnionFind()
    for x, y in pairs:
        uf.union(image_id_to_idx[x], image_id_to_idx[y])

    idx2cluster = [uf.find(i) for i in range(len(records))]
    cluster2idx = defaultdict(set)
    for i, x in enumerate(idx2cluster):
        cluster2idx[x].add(i)

    for i in range(len(records)):
        records[i]["cluster"] = idx2cluster[i]
        records[i]["duplicates"] = list(cluster2idx[idx2cluster[i]])

    return datasets.Dataset.from_list(records)


def preprocess_csv(file, path, split):
    with ZipFile(file, "r") as zip:
        df = pd.read_csv(zip.open(path), sep="\t").drop(columns=["Unnamed: 0"])
        df = df.assign(split=split)
        return datasets.Dataset.from_pandas(df)


if __name__ == "__main__":
    test = preprocess_split("./data/test_sets.zip", "test_inf_data.json", "full_test_gt.json")
    val = preprocess_split("./data/evaluation_set.zip", "1955_inf_data.json", "1955_gt.json")
    # train = preprocess_split("./data/training_set.zip", "1948_inf_data.json", "1948_gt.json")
    train = preprocess_csv(
        "./data/training_sets.zip",
        "train_set.csv",
        split="train",
    )
    dev = preprocess_csv(
        "./data/training_sets.zip",
        "dev_set.csv",
        split="dev",
    )

    datasets.DatasetDict({"train": train, "dev": dev}).push_to_hub("chenghao/NEWS-COPY-train")
    datasets.DatasetDict({"test": test, "val": val}).push_to_hub("chenghao/NEWS-COPY-eval")
