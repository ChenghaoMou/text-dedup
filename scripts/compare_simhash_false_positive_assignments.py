#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import cast

import polars as pl
from datasets import Dataset

from text_dedup.config import SimHashAlgorithmConfig
from text_dedup.config.base import load_config_from_toml
from text_dedup.simhash import assign
from text_dedup.simhash import cluster
from text_dedup.simhash import fingerprint
from text_dedup.simhash import load_and_preprocess
from text_dedup.utils.jaccard import jaccard_similarity
from text_dedup.utils.union_find import UnionFind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SimHash false-positive rates before and after root injection / duplicate refresh."
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/config_simhash_test.toml"), help="Path to TOML config file.")
    parser.add_argument("--limit", type=int, default=20, help="How many changed rows to print.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save the full diff as CSV.")
    return parser.parse_args()


def build_cluster_groups(
    algo: SimHashAlgorithmConfig,
    ds: Dataset,
    ds_candidates: Dataset,
    *,
    include_roots: bool,
) -> dict[int, list[tuple[int, str]]]:
    cluster_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for record in ds_candidates:
        cluster_id = record[algo.cluster_column]
        idx = record[algo.internal_index_column]
        text = record[algo.text_column]
        cluster_groups[cluster_id].append((idx, text))

    if include_roots:
        for root_id in sorted(cluster_groups.keys()):
            record = ds[root_id]
            cluster_groups[root_id].insert(0, (root_id, record[algo.text_column]))

    return cluster_groups


def verify_assignments(
    config,
    ds: Dataset,
    *,
    include_roots: bool,
    refresh_duplicate_flag: bool,
) -> tuple[Dataset, dict[int, int], dict[str, int]]:
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    ds_candidates = ds.filter(
        function=lambda x: x["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
        num_proc=algo.num_proc,
    )

    if len(ds_candidates) == 0:
        return ds, {}, {"candidate_docs": 0, "candidate_clusters": 0, "verified_pairs": 0, "true_pairs": 0}

    cluster_groups = build_cluster_groups(algo, ds, ds_candidates, include_roots=include_roots)
    tokenizer = algo.get_ngrams_func()
    verified_pairs = 0
    true_pairs: list[tuple[int, int]] = []

    for members in cluster_groups.values():
        if len(members) < 2:
            continue

        for i, (idx1, text1) in enumerate(members):
            tokens1 = set(tokenizer(text1))
            for j in range(i + 1, len(members)):
                idx2, text2 = members[j]
                verified_pairs += 1

                tokens2 = set(tokenizer(text2))
                similarity = jaccard_similarity(tokens1, tokens2)
                if similarity >= algo.jaccard_threshold:
                    true_pairs.append((idx1, idx2))

    uf = UnionFind()
    for idx1, idx2 in true_pairs:
        uf.union(idx1, idx2)

    new_parents = {k: v for k, v in uf.get_clusters().items() if k != v}

    if refresh_duplicate_flag:
        updated_ds = ds.map(
            function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
                algo.cluster_column: new_parents.get(
                    record[algo.internal_index_column],
                    record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
                ),
                "__duplicate__": record[algo.internal_index_column] in new_parents,
            },
            with_indices=False,
            num_proc=algo.num_proc,
            desc="Updating clusters...",
        )
    else:
        updated_ds = ds.map(
            function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
                algo.cluster_column: new_parents.get(
                    record[algo.internal_index_column],
                    record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
                )
            },
            with_indices=False,
            num_proc=algo.num_proc,
            desc="Updating clusters...",
        )

    stats = {
        "candidate_docs": len(ds_candidates),
        "candidate_clusters": len(cluster_groups),
        "verified_pairs": verified_pairs,
        "true_pairs": len(true_pairs),
    }
    return updated_ds, new_parents, stats


def main() -> None:
    args = parse_args()
    config = load_config_from_toml(args.config)

    if not isinstance(config.algorithm, SimHashAlgorithmConfig):
        raise ValueError("This script only supports SimHash configs.")  # noqa: TRY003

    algo = cast(SimHashAlgorithmConfig, config.algorithm)

    ds, original_len = load_and_preprocess(config)
    embedded = fingerprint(config, ds)
    parents = cluster(config, embedded)
    ds = assign(config, ds, parents)

    old_ds, old_parents, old_stats = verify_assignments(
        config,
        ds,
        include_roots=False,
        refresh_duplicate_flag=False,
    )
    new_ds, new_parents, new_stats = verify_assignments(
        config,
        ds,
        include_roots=True,
        refresh_duplicate_flag=True,
    )

    comparison = pl.DataFrame({
        "id": new_ds[algo.internal_index_column],
        "cluster_old": old_ds[algo.cluster_column],
        "cluster_new": new_ds[algo.cluster_column],
        "duplicate_old": old_ds["__duplicate__"],
        "duplicate_new": new_ds["__duplicate__"],
    }).sort("id")

    cluster_diff = comparison.filter(pl.col("cluster_old") != pl.col("cluster_new"))
    duplicate_diff = comparison.filter(pl.col("duplicate_old") != pl.col("duplicate_new"))
    diff = comparison.filter(
        (pl.col("cluster_old") != pl.col("cluster_new")) | (pl.col("duplicate_old") != pl.col("duplicate_new"))
    )
    candidate_docs = old_stats["candidate_docs"]
    old_false_positives = candidate_docs - len(old_parents)
    new_false_positives = candidate_docs - len(new_parents)
    old_false_positive_rate = old_false_positives / candidate_docs if candidate_docs else 0.0
    new_false_positive_rate = new_false_positives / candidate_docs if candidate_docs else 0.0

    print(f"Config path              : {args.config}")
    print(f"Original docs            : {original_len}")
    print(f"Candidate docs           : {candidate_docs}")
    print(f"Initial candidate groups : {old_stats['candidate_clusters']}")
    print(f"Old verified pairs       : {old_stats['verified_pairs']}")
    print(f"New verified pairs       : {new_stats['verified_pairs']}")
    print(f"Old assignments          : {len(old_parents)}")
    print(f"New assignments          : {len(new_parents)}")
    print(f"Old false positives      : {old_false_positives}")
    print(f"New false positives      : {new_false_positives}")
    print(f"Old false positive rate  : {old_false_positive_rate:.2%}")
    print(f"New false positive rate  : {new_false_positive_rate:.2%}")
    print(f"Changed cluster ids      : {len(cluster_diff)}")
    print(f"Changed duplicate flags  : {len(duplicate_diff)}")
    print(f"Changed rows overall     : {len(diff)}")

    if len(diff) > 0:
        print(f"\nTop {min(args.limit, len(diff))} changed rows:")
        print(diff.head(args.limit))
    else:
        print("\nNo differences found between old and new verification behavior.")

    if args.output_csv is not None:
        diff.write_csv(args.output_csv)
        print(f"\nFull diff written to: {args.output_csv}")


if __name__ == "__main__":
    main()
