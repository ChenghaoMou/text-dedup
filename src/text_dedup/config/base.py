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
