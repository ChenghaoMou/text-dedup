from typing import Literal
from typing import TypeAlias

from pydantic_settings import BaseSettings


class OutputConfig(BaseSettings):
    output_type: Literal["hf"]
    output_dir: str
    skip_filtering: bool = False
    clean_cache: bool = False
    save_clusters: bool = False
    keep_index_column: bool = False
    keep_cluster_column: bool = False


class HFOutputConfig(OutputConfig):
    output_type: Literal["hf"] = "hf"


OutputConfigType: TypeAlias = HFOutputConfig
