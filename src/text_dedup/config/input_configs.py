from typing import Literal
from typing import TypeAlias

from pydantic_settings import BaseSettings


class InputConfig(BaseSettings):
    input_type: Literal["hf", "local_jsonl"]


class HFInputConfig(InputConfig):
    input_type: Literal["hf"] = "hf"
    path: str
    name: str | None = None
    data_dir: str | None = None
    data_files: list[str] | None = None
    split: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = False


class JSONLInputConfig(InputConfig):
    input_type: Literal["local_jsonl"] = "local_jsonl"
    path: str


InputConfigType: TypeAlias = HFInputConfig | JSONLInputConfig
