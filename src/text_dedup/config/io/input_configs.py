from typing import Any
from typing import Literal

from pydantic_settings import BaseSettings


class InputConfig(BaseSettings):
    input_type: Literal["local_files", "local_hf_dataset"]


class LocalInputConfig(InputConfig):
    input_type: Literal["local_files"] = "local_files"  # pyright: ignore[reportIncompatibleVariableOverride]
    file_type: Literal["parquet", "csv", "json"] = "parquet"
    read_arguments: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class LocalHFDatasetInputConfig(InputConfig):
    input_type: Literal["local_hf_dataset"] = "local_hf_dataset"  # pyright: ignore[reportIncompatibleVariableOverride]
    read_arguments: dict[str, Any]  # pyright: ignore[reportExplicitAny]


type InputConfigType = LocalInputConfig | LocalHFDatasetInputConfig
