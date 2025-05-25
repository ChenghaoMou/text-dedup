import os
from typing import Any

from datasets import Dataset
from datasets import load_dataset as load_dataset_hf

from text_dedup.config.base import Config
from text_dedup.config.base import HFInputConfig
from text_dedup.config.base import HFOutputConfig
from text_dedup.utils.union_find import UnionFind


def load_dataset(config: Config) -> Dataset:
    match config.input:
        case HFInputConfig():
            _INTERNAL_INDEX_COLUMN = config.algorithm.internal_index_column
            result = load_dataset_hf(**config.input.model_dump(exclude={"input_type"}))
        case _:
            raise ValueError(f"Unsupported input type: {config.input}")  # noqa: TRY003

    if not isinstance(result, Dataset):
        raise TypeError(f"Expected Dataset, got {type(result)}")  # noqa: TRY003

    ds: Dataset = result
    ds = ds.map(lambda _, i: {_INTERNAL_INDEX_COLUMN: i}, with_indices=True, num_proc=config.algorithm.num_proc)
    return ds


def save_dataset(config: Config, *, final_data: Dataset, uf: UnionFind[Any] | None = None, **kwargs: Any) -> None:
    match config.output:
        case HFOutputConfig():
            columns_to_remove = {
                config.algorithm.internal_index_column,
                config.algorithm.cluster_column,
            }
            if config.output.keep_index_column:
                columns_to_remove.remove(config.algorithm.internal_index_column)
            if config.output.keep_cluster_column:
                columns_to_remove.remove(config.algorithm.cluster_column)
            if columns_to_remove:
                final_data = final_data.remove_columns(list(columns_to_remove))
            final_data.save_to_disk(config.output.output_dir)

    if config.output.save_clusters and uf is not None:
        uf.dump(os.path.join(config.output.output_dir, "clusters.pickle"))
