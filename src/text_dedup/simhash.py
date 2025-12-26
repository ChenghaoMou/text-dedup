from typing import cast

import polars as pl
from bitarray import frozenbitarray
from datasets import Dataset
from polars.dataframe import DataFrame
from polars_grouper import super_merger

from text_dedup.config import Config
from text_dedup.config import SimHashAlgorithmConfig
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset
from text_dedup.utils.jaccard import jaccard_similarity
from text_dedup.utils.logger import log
from text_dedup.utils.timer import Timer


def load_and_preprocess(config: Config) -> tuple[Dataset, int]:
    """Load and preprocess the dataset."""
    ds = load_dataset(config)
    original_len = len(ds)
    return ds, original_len


def fingerprint(config: Config, ds: Dataset) -> Dataset:
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    embedded: Dataset = ds.map(
        function=algo.get_embed_func(),
        input_columns=[algo.text_column, algo.internal_index_column],
        remove_columns=[col for col in ds.column_names if col != algo.internal_index_column],
        num_proc=algo.num_proc,
        with_indices=False,
        batched=True,
        batch_size=1,
        # new_fingerprint=str(uuid.uuid4()),
        desc="SimHashing...",
    )
    return embedded


def cluster(config: Config, ds: Dataset) -> dict[int, int]:
    """Cluster the dataset."""
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    signatures: DataFrame = ds.select_columns(["__key__", "__val__", algo.internal_index_column]).to_polars()  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
    clusters = (
        signatures.group_by(["__key__"])
        .agg(items=pl.struct(["__val__", algo.internal_index_column]))
        .filter(pl.col("items").list.len() > 1)
        .with_row_index(name="index")
        .drop("__key__")
    )

    exploded_df = clusters.explode("items")
    combinations = (
        exploded_df.join(exploded_df, on="index")
        .filter(
            pl.col("items").struct.field(algo.internal_index_column)
            < pl.col("items_right").struct.field(algo.internal_index_column)
        )
        .with_columns(
            hamming_score=pl.struct(pl.all()).map_elements(
                lambda record: algo.hamming_distance(
                    frozenbitarray(record["items"]["__val__"]),
                    frozenbitarray(record["items_right"]["__val__"]),
                ),
                return_dtype=pl.Int64,
            )
        )
        .filter(pl.col("hamming_score") <= algo.bit_diff)
        .select([
            pl.col("items").struct.field(algo.internal_index_column).alias("src"),
            pl.col("items_right").struct.field(algo.internal_index_column).alias("dst"),
        ])
        .unique()
    )

    grouped = super_merger(combinations, from_col_name="src", to_col_name="dst")
    mapping = pl.concat([
        grouped.select(pl.col("src").alias("id"), pl.col("group").alias("cluster")).unique(),
        grouped.select(pl.col("dst").alias("id"), pl.col("group").alias("cluster")).unique(),
    ]).unique()

    return dict(mapping.iter_rows(named=False))


def assign(config: Config, ds: Dataset, parents: dict[int, int]) -> Dataset:
    """Assign cluster id to the dataset."""
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    ds = ds.map(
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            algo.cluster_column: parents.get(
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
            ),
            "__duplicate__": record[algo.internal_index_column] in parents,
        },
        with_indices=False,
        # ! parents is pickled to multiple processes
        num_proc=config.algorithm.num_proc,
        desc="Assigning initial clusters...",
    )
    return ds


def check_false_positives(config: Config, ds: Dataset) -> tuple[Dataset, dict[int, int]]:
    """Check false positives."""
    algo = cast(SimHashAlgorithmConfig, config.algorithm)
    ds_candidates = ds.filter(
        function=lambda x: x["__duplicate__"],  # pyright: ignore[reportUnknownLambdaType]
        num_proc=algo.num_proc,
    )
    candidates: DataFrame = ds_candidates.select_columns([  # pyright: ignore[reportAssignmentType]
        algo.internal_index_column,
        algo.text_column,
        algo.cluster_column,
    ]).to_polars()
    cluster_num = candidates.unique(algo.cluster_column).shape[0]
    candidates = candidates.join(candidates, on=algo.cluster_column).filter(
        pl.col(algo.internal_index_column) < pl.col(f"{algo.internal_index_column}_right")
    )
    verified_pairs = len(candidates)
    tokenizer = algo.get_ngrams_func()
    results = (
        candidates.with_columns(
            jaccard_score=pl.struct(pl.all()).map_elements(
                lambda record: jaccard_similarity(  # pyright: ignore[reportAny]
                    set(tokenizer(record[algo.text_column])),  # pyright: ignore[reportAny]
                    set(tokenizer(record[f"{algo.text_column}_right"])),  # pyright: ignore[reportAny]
                ),
                return_dtype=pl.Float64,
            )
        )
        .filter(pl.col("jaccard_score") >= algo.jaccard_threshold)
        .with_columns([
            pl.col(algo.internal_index_column).alias("idx1"),
            pl.col(f"{algo.internal_index_column}_right").alias("idx2"),
        ])
    )

    assignment = (
        results.select([pl.col("idx1").alias("idx"), pl.col(algo.cluster_column)])
        .vstack(results.select([pl.col("idx2").alias("idx"), pl.col(algo.cluster_column)]))
        .unique()
        # update the cluster id to the minimum index
        .group_by(algo.cluster_column)
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

    ds = ds.map(
        function=lambda record: {  # pyright: ignore[reportUnknownLambdaType]
            algo.cluster_column: new_parents.get(
                record[algo.internal_index_column],
                record[algo.internal_index_column],  # pyright: ignore[reportUnknownArgumentType]
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
    if not config.output.skip_filtering:
        result: Dataset = ds.filter(
            function=lambda record: not record["duplicate"],  # pyright: ignore[reportUnknownLambdaType]
            with_indices=False,
            num_proc=config.algorithm.num_proc,
            desc="Removing duplicates...",
        )
        return result
    return ds


def main(config: Config) -> None:
    """
    Running BloomFilter algorithm.

    Parameters
    ----------
    config: Config
        The deduplication configuration object.

    """

    timer = Timer()
    algo = cast(SimHashAlgorithmConfig, config.algorithm)

    with timer("Total", enable_spin=False):
        with timer("Preprocessing", enable_spin=False):
            ds, ORIGINAL_LEN = load_and_preprocess(config)

        with timer("SimHashing", enable_spin=False):
            embedded = fingerprint(config, ds)

        with timer("Clustering"):
            assignment = cluster(config, embedded)
            ds = assign(config, ds, assignment)

        with timer("Verifying"):
            if algo.check_false_positive:
                ds, assignment = check_false_positives(config, ds)

        with timer("Filtering", enable_spin=False):
            final_data = remove_duplicates(config, ds)

        with timer("Saving"):
            save_dataset(config, final_data=final_data, clusters=assignment)

        with timer("Cleaning"):
            if config.output.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

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
