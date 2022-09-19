from typing import Any
from typing import Dict
from typing import Optional

from text_dedup.utils.storage.base import StorageDict
from text_dedup.utils.storage.mem_dict import MemDict
from text_dedup.utils.storage.redis_dict import RedisDict


def create_storage(storage_config: Optional[Dict[str, Any]] = None) -> StorageDict:
    """
    Create a storage object based on the storage configuration.

    Parameters
    ----------
    storage_config : Dict[str, Any]
        Storage configuration.

    Returns
    -------
    StorageDict
        A storage object.

    Raises
    ------
    ValueError
        If the storage type is not supported.
    """
    storage_config = storage_config or {}
    if storage_config.get("type", None) == "redis":
        return RedisDict(storage_config)

    return MemDict(storage_config)


__all__ = ["create_storage", "StorageDict"]
