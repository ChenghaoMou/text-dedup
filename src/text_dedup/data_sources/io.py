import pickle
from pathlib import Path
from typing import Any
from typing import cast

from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from datasets import disable_progress_bars  # pyright: ignore[reportMissingTypeStubs]
from datasets import enable_progress_bars  # pyright: ignore[reportMissingTypeStubs]
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset as hf_load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_from_disk as hf_load_from_disk,  # pyright: ignore[reportUnknownVariableType]
)

from text_dedup.config import Config
from text_dedup.config.io import LocalInputConfig
from text_dedup.config.io import OutputConfig
from text_dedup.config.io.input_configs import LocalHFDatasetInputConfig
from text_dedup.utils.logger import log
from text_dedup.utils.progress import use_tqdm


class InvalidDatasetTypeError(Exception):
    def __init__(self, data_type: type) -> None:
        super().__init__(f"Expecting Dataset object, loaded {data_type} instead")


def load_dataset(config: Config) -> Dataset:
    match config.input:
        case LocalHFDatasetInputConfig():
            disable_progress_bars()
            with use_tqdm():
                ds = hf_load_from_disk(**config.input.read_arguments)  # pyright: ignore[reportAny]
            enable_progress_bars()
            if not isinstance(ds, Dataset):
                raise InvalidDatasetTypeError(type(ds))
            _INTERNAL_INDEX_COLUMN = config.algorithm.internal_index_column
            ds = ds.map(  # pyright: ignore[reportUnknownMemberType]
                lambda _, i: {_INTERNAL_INDEX_COLUMN: i},  # pyright: ignore[reportUnknownLambdaType]
                with_indices=True,
                num_proc=config.algorithm.num_proc,
                desc="Indexing",
            )
            return cast(Dataset, ds)

        case LocalInputConfig():
            _INTERNAL_INDEX_COLUMN = config.algorithm.internal_index_column

            disable_progress_bars()
            with use_tqdm():
                ds = hf_load_dataset(**config.input.read_arguments)  # pyright: ignore[reportAny]
            enable_progress_bars()

            if not isinstance(ds, Dataset):
                raise InvalidDatasetTypeError(type(ds))
            ds = ds.map(  # pyright: ignore[reportUnknownMemberType]
                lambda _, i: {_INTERNAL_INDEX_COLUMN: i},  # pyright: ignore[reportUnknownLambdaType]
                with_indices=True,
                num_proc=config.algorithm.num_proc,
                desc="Indexing",
            )
            return cast(Dataset, ds)


def save_dataset(config: Config, *, final_data: Dataset, clusters: dict[int, int], **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny, reportUnusedParameter]
    """Save the dataset to disk."""
    # Create output directory if it doesn't exist
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.save_clusters:
        if not config.output.keep_index_column:
            log.warning("Saving clusters requires `--keep-index-column`, turning it on")
            config.output.keep_index_column = True
        with open(output_dir / "clusters.pickle", "wb") as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    match config.output:
        case OutputConfig():
            columns_to_remove = {
                config.algorithm.internal_index_column,
                config.algorithm.cluster_column,
            }
            if config.output.keep_index_column and config.algorithm.internal_index_column in columns_to_remove:
                columns_to_remove.remove(config.algorithm.internal_index_column)
            if (
                config.output.keep_cluster_column or config.output.save_clusters
            ) and config.algorithm.cluster_column in columns_to_remove:
                columns_to_remove.remove(config.algorithm.cluster_column)
            if columns_to_remove:
                columns_to_remove = [col for col in columns_to_remove if col in final_data.column_names]
                final_data = final_data.remove_columns(list(columns_to_remove))

            final_data.save_to_disk(config.output.output_dir, num_proc=config.algorithm.num_proc)  # pyright: ignore[reportUnknownMemberType]
