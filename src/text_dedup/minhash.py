# pyright: reportMissingTypeStubs=false
from itertools import combinations
from typing import cast

import polars as pl
from datasets import Dataset
from polars.dataframe import DataFrame
from polars_grouper import super_merger

from text_dedup.config import Config
from text_dedup.config import MinHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.jaccard import jaccard_similarity
from text_dedup.utils.logger import log
from text_dedup.utils.timer import Timer


def load_and_preprocess(config: Config) -> tuple[Dataset, int, int]:
    """Load and preprocess the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ds = load_dataset(config)
    original_len = len(ds)
    result: Dataset = ds.filter(  # pyright: ignore[reportUnknownMemberType]
        minhash_args.get_filtering_func(),
        input_columns=[minhash_args.text_column],
        num_proc=config.algorithm.num_proc,
        desc="Filtering...",
    )
    filtered_len = len(result)
    return result, original_len, filtered_len


def embed(config: Config, ds: Dataset) -> Dataset:
    """Fingerprint the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    result: Dataset = ds.map(  # pyright: ignore[reportUnknownMemberType]
        function=minhash_args.get_embed_func(),
        input_columns=[minhash_args.text_column, minhash_args.internal_index_column],
        remove_columns=[col for col in ds.column_names if col != minhash_args.internal_index_column],
        num_proc=config.algorithm.num_proc,
        batched=True,
        batch_size=1,
        desc="Fingerprinting...",
    )
    return result


def clustering(config: Config, ds: Dataset) -> dict[int, int]:
    """Cluster the dataset."""
    algo = cast(MinHashAlgorithmConfig, config.algorithm)
    signatures: DataFrame = ds.select_columns(["__band_idx__", "__band_val__", algo.internal_index_column]).to_polars()  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
    clusters = (
        signatures.group_by(["__band_idx__", "__band_val__"])
        .agg(pl.col(algo.internal_index_column))
        .filter(pl.col(algo.internal_index_column).list.len() > 1)
        .select(pl.col(algo.internal_index_column).alias("values"))
        .with_row_index(name="index")
    )
    exploded_df = clusters.explode("values")
    combinations = (
        exploded_df.join(exploded_df, on="index")
        .filter(pl.col("values") < pl.col("values_right"))
        .rename({"values": "src", "values_right": "dst"})
        .drop("index")
    )
    grouped = super_merger(combinations, from_col_name="src", to_col_name="dst")
    mapping = pl.concat([
        grouped.select(pl.col("src").alias("id"), pl.col("group").alias("cluster")).unique(),
        grouped.select(pl.col("dst").alias("id"), pl.col("group").alias("cluster")).unique(),
    ]).unique()

    return dict(mapping.iter_rows(named=False))


def assign_clusters(config: Config, ds: Dataset, parents: dict[int, int]) -> Dataset:
    """Assign cluster id to the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ds = ds.map(  # pyright: ignore[reportUnknownMemberType]
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            minhash_args.cluster_column: parents.get(
                record[minhash_args.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
                record[minhash_args.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
            ),
            "__duplicate__": record[minhash_args.internal_index_column] in parents,
        },
        with_indices=False,
        # ! parents is pickled to multiple processes
        num_proc=config.algorithm.num_proc,
        desc="Assigning initial clusters...",
    )
    return ds


def check_false_positives(config: Config, ds: Dataset) -> tuple[Dataset, dict[int, int]]:
    """Check false positives."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    ds_candidates = ds.filter(  # pyright: ignore[reportUnknownMemberType]
        function=lambda x: x["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
        num_proc=minhash_args.num_proc,
    )
    candidates = pl.from_dict({  # pyright: ignore[reportUnknownArgumentType]
        minhash_args.internal_index_column: ds_candidates[minhash_args.internal_index_column],
        minhash_args.text_column: ds_candidates[minhash_args.text_column],
        minhash_args.cluster_column: ds_candidates[minhash_args.cluster_column],
    })
    cluster_num = candidates.unique(minhash_args.cluster_column).shape[0]
    candidates = (
        candidates.group_by(pl.col(minhash_args.cluster_column))
        .agg([
            pl.col(minhash_args.internal_index_column),
            pl.col(minhash_args.text_column),
        ])
        .with_columns(
            pairs=pl.struct([
                pl.col(minhash_args.internal_index_column),
                pl.col(minhash_args.text_column),
            ]).map_elements(
                lambda x: [  # pyright: ignore[reportAny]
                    {"pair1": {"idx": p1[0], "text": p1[1]}, "pair2": {"idx": p2[0], "text": p2[1]}}
                    for (p1, p2) in combinations(
                        zip(x[minhash_args.internal_index_column], x[minhash_args.text_column], strict=True),  # pyright: ignore[reportAny]
                        2,
                    )
                ],
                return_dtype=pl.List(
                    pl.Struct({
                        "pair1": pl.Struct({"idx": pl.Int64, "text": pl.String}),
                        "pair2": pl.Struct({"idx": pl.Int64, "text": pl.String}),
                    })
                ),
            )
        )
        .select([minhash_args.cluster_column, "pairs"])
        .explode("pairs")
    )
    verified_pairs = len(candidates)
    tokenizer = minhash_args.get_ngrams_func()
    results = candidates.with_columns(
        jaccard_score=pl.col("pairs").map_elements(
            lambda pair: jaccard_similarity(  # pyright: ignore[reportAny]
                set(tokenizer(pair["pair1"]["text"])),  # pyright: ignore[reportAny]
                set(tokenizer(pair["pair2"]["text"])),  # pyright: ignore[reportAny]
            ),
            return_dtype=pl.Float64,
        )
    ).filter(pl.col("jaccard_score") >= minhash_args.threshold)

    assignment = (
        results.with_columns([
            pl.col("pairs").struct.field("pair1").struct.field("idx").alias("idx1"),
            pl.col("pairs").struct.field("pair2").struct.field("idx").alias("idx2"),
        ])
        .select([pl.col("idx1").alias("idx"), pl.col(minhash_args.cluster_column)])
        .vstack(
            results.with_columns([
                pl.col("pairs").struct.field("pair1").struct.field("idx").alias("idx1"),
                pl.col("pairs").struct.field("pair2").struct.field("idx").alias("idx2"),
            ]).select([pl.col("idx2").alias("idx"), pl.col(minhash_args.cluster_column)])
        )
        .unique()
        # update the cluster id to the minimum index
        .group_by(minhash_args.cluster_column)
        .agg(pl.col("idx"), pl.min("idx").alias("cluster_id"))
        .select([pl.col("idx"), pl.col("cluster_id")])
        .explode("idx")
    )

    new_parents = dict(assignment.iter_rows(named=False))
    total_true_positives = len(new_parents)
    total_false_positives = len(ds_candidates) - total_true_positives
    total_true_positive_clusters = len(set(new_parents.values()))

    log.info(f"Verified {cluster_num} clusters/{verified_pairs} pairs")
    log.info(f"False Positives   : {total_false_positives}")
    log.info(f"True Positives    : {total_true_positives}")
    log.info(f"True Clusters     : {total_true_positive_clusters}")

    ds = ds.map(  # pyright: ignore[reportUnknownMemberType]
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            minhash_args.cluster_column: new_parents.get(
                record[minhash_args.internal_index_column],
                record[minhash_args.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
            )
        },
        with_indices=False,
        # ! new_parents is pickled
        num_proc=config.algorithm.num_proc,
        desc="Updating clusters...",
    )

    return ds, new_parents


def remove_duplicates(config: Config, ds: Dataset) -> Dataset:
    """Remove duplicates from the dataset."""
    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    if not config.output.skip_filtering:
        result: Dataset = ds.filter(  # pyright: ignore[reportUnknownMemberType]
            function=lambda record: record[minhash_args.cluster_column] == record[minhash_args.internal_index_column],  # pyright: ignore[reportUnknownLambdaType]
            with_indices=False,
            num_proc=config.algorithm.num_proc,
            desc="Removing duplicates...",
        )
        return result
    return ds


def main(config: Config) -> None:
    """
    Running MinHash algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    minhash_args = cast(MinHashAlgorithmConfig, config.algorithm)
    timer = Timer()

    with timer("Total", enable_spin=False):
        with timer("Preprocessing", enable_spin=False):
            ds, ORIGINAL_LEN, FILTERED_LEN = load_and_preprocess(config)
            log.info(f"Filtered {ORIGINAL_LEN - FILTERED_LEN} records")

        with timer("MinHashing", enable_spin=False):
            embedded = embed(config, ds)

        with timer("Clustering"):
            assignment = clustering(config, embedded)
            ds = assign_clusters(config, ds, assignment)

        with timer("Verifying"):
            if minhash_args.check_false_positive:
                ds, assignment = check_false_positives(config, ds)

        with timer("Filtering", enable_spin=False):
            final_data = remove_duplicates(config, ds)

        with timer("Saving"):
            save_dataset(config, final_data=final_data, clusters=assignment)

        with timer("Cleaning"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()  # pyright: ignore[reportUnusedCallResult]
                final_data.cleanup_cache_files()  # pyright: ignore[reportUnusedCallResult]

    timer.report({"Before": ORIGINAL_LEN, "After": len(final_data)})


if __name__ == "__main__":
    from pydantic_settings import CliApp

    from text_dedup.config.base import Config
    from text_dedup.utils.env import check_env
    from text_dedup.utils.progress import use_custom_progress_bar

    with use_custom_progress_bar():
        config = CliApp.run(Config)
        check_env()
        if config.debug.enable_profiling:
            from scalene.scalene_profiler import enable_profiling

            with enable_profiling():
                main(config)
        else:
            main(config)
