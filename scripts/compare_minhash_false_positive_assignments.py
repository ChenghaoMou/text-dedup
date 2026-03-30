#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import polars as pl
from polars_grouper import super_merger

from text_dedup.config import MinHashAlgorithmConfig
from text_dedup.config.base import load_config_from_toml
from text_dedup.minhash import assign
from text_dedup.minhash import cluster
from text_dedup.minhash import fingerprint
from text_dedup.minhash import load_and_preprocess
from text_dedup.utils.jaccard import jaccard_similarity


def empty_assignment(column_name: str) -> pl.DataFrame:
    return pl.DataFrame(schema={"id": pl.Int64, column_name: pl.Int64})


def build_old_assignment(results: pl.DataFrame, cluster_column: str) -> pl.DataFrame:
    if results.is_empty():
        return empty_assignment("cluster_old")

    return (
        results.select([pl.col("idx1").alias("id"), pl.col(cluster_column)])
        .vstack(results.select([pl.col("idx2").alias("id"), pl.col(cluster_column)]))
        .unique()
        .group_by(cluster_column)
        .agg(pl.col("id"), pl.min("id").alias("cluster_old"))
        .select(pl.col("id"), pl.col("cluster_old"))
        .explode("id")
        .sort("id")
    )


def build_new_assignment(results: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if results.is_empty():
        return empty_assignment("cluster_new"), results.with_columns(pl.lit(None, dtype=pl.UInt64).alias("group"))

    super_merger_results = super_merger(results, from_col_name="idx1", to_col_name="idx2")
    new_assignment = (
        pl.concat([
            super_merger_results.select(pl.col("idx1").alias("id"), pl.col("group")).unique(),
            super_merger_results.select(pl.col("idx2").alias("id"), pl.col("group")).unique(),
        ])
        .unique()
        .group_by("group")
        .agg(pl.col("id"), pl.min("id").alias("cluster_new"))
        .select(pl.col("id"), pl.col("cluster_new"))
        .explode("id")
        .sort("id")
    )
    return new_assignment, super_merger_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MinHash false-positive assignments before and after connected-components regrouping."
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/config_minhash_test.toml"), help="Path to TOML config file.")
    parser.add_argument("--limit", type=int, default=20, help="How many diff rows to print.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save the full diff as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_toml(args.config)

    if not isinstance(config.algorithm, MinHashAlgorithmConfig):
        raise ValueError("This script only supports MinHash configs.")  # noqa: TRY003

    algo = cast(MinHashAlgorithmConfig, config.algorithm)

    ds, original_len, filtered_len = load_and_preprocess(config)
    embedded = fingerprint(config, ds)
    parents = cluster(config, embedded)
    ds = assign(config, ds, parents)

    ds_candidates = ds.filter(  # pyright: ignore[reportUnknownMemberType]
        function=lambda x: x["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
        num_proc=algo.num_proc,
    )
    candidates = ds_candidates.select_columns([  # pyright: ignore[reportUnknownMemberType]
        algo.internal_index_column,
        algo.text_column,
        algo.cluster_column,
    ]).to_polars()

    candidate_cluster_count = candidates.unique(algo.cluster_column).shape[0]
    pair_candidates = candidates.join(candidates, on=algo.cluster_column).filter(
        pl.col(algo.internal_index_column) < pl.col(f"{algo.internal_index_column}_right")
    )

    tokenizer = algo.get_ngrams_func()
    results = (
        pair_candidates.with_columns(
            jaccard_score=pl.struct(pl.all()).map_elements(
                lambda record: jaccard_similarity(
                    set(tokenizer(record[algo.text_column])),
                    set(tokenizer(record[f"{algo.text_column}_right"])),
                ),
                return_dtype=pl.Float64,
            )
        )
        .filter(pl.col("jaccard_score") >= algo.threshold)
        .with_columns([
            pl.col(algo.internal_index_column).alias("idx1"),
            pl.col(f"{algo.internal_index_column}_right").alias("idx2"),
        ])
    )

    old_assignment = build_old_assignment(results, algo.cluster_column)
    new_assignment, super_merger_results = build_new_assignment(results)

    diff = (
        old_assignment.join(new_assignment, on="id", how="inner")
        .filter(pl.col("cluster_old") != pl.col("cluster_new"))
        .sort("id")
    )

    split_clusters = (
        super_merger_results.group_by(algo.cluster_column)
        .agg(pl.col("group").n_unique().alias("new_group_count"))
        .filter(pl.col("new_group_count") > 1)
        .sort("new_group_count", descending=True)
        if not results.is_empty()
        else pl.DataFrame(schema={algo.cluster_column: pl.Int64, "new_group_count": pl.UInt32})
    )

    print(f"Config path              : {args.config}")
    print(f"Original docs            : {original_len}")
    print(f"Filtered docs            : {filtered_len}")
    print(f"Candidate docs           : {len(ds_candidates)}")
    print(f"Candidate clusters       : {candidate_cluster_count}")
    print(f"Candidate pairs          : {len(pair_candidates)}")
    print(f"Verified pairs           : {len(results)}")
    print(f"Old assignment rows      : {len(old_assignment)}")
    print(f"New assignment rows      : {len(new_assignment)}")
    print(f"Changed cluster ids      : {len(diff)}")
    print(f"Split candidate clusters : {len(split_clusters)}")

    if len(diff) > 0:
        print(f"\nTop {min(args.limit, len(diff))} changed rows:")
        print(diff.head(args.limit))
    else:
        print("\nNo cluster_id differences found.")

    if len(split_clusters) > 0:
        print(f"\nTop {min(args.limit, len(split_clusters))} split candidate clusters:")
        print(split_clusters.head(args.limit))

    if args.output_csv is not None:
        diff.write_csv(args.output_csv)
        print(f"\nFull diff written to: {args.output_csv}")


if __name__ == "__main__":
    main()
