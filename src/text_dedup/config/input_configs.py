from typing import Any
from typing import Literal
from typing import TypeAlias

from pydantic_settings import BaseSettings


class InputConfig(BaseSettings):
    input_type: Literal["local_files"]


class LocalInputConfig(InputConfig):
    input_type: Literal["local_files"] = "local_files"
    file_type: Literal["parquet", "csv", "json"] = "parquet"
    read_arguments: dict[str, Any]  # pyright: ignore[reportExplicitAny]


InputConfigType: TypeAlias = LocalInputConfig
