from pathlib import Path
from typing import Any
from typing import override

from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import SettingsConfigDict
from pydantic_settings import TomlConfigSettingsSource

from .algorithms import AlgoConfig
from .debug import DebugConfig
from .io import InputConfigType
from .io import OutputConfigType


class Config(BaseSettings):
    input: InputConfigType
    algorithm: AlgoConfig
    output: OutputConfigType
    debug: DebugConfig

    model_config = SettingsConfigDict(toml_file="config.toml")  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any) -> None:  # pragma: no cover
        from .algorithms import SuffixArrayAlgorithmConfig

        super().model_post_init(context)
        if isinstance(self.algorithm, SuffixArrayAlgorithmConfig):
            self.output.save_clusters = False
            self.output.keep_cluster_column = False

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # pragma: no cover
        return (TomlConfigSettingsSource(settings_cls),)


def load_config_from_toml(toml_path: Path) -> Config:  # pragma: no cover
    """Load Config from a TOML file.

    Parameters
    ----------
    toml_path : Path
        Path to the TOML configuration file

    Returns
    -------
    Config
        Loaded configuration object
    """
    original_config = Config.model_config.copy()
    Config.model_config = SettingsConfigDict(toml_file=str(toml_path))
    try:
        config = Config()  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    finally:
        Config.model_config = original_config
    return config
