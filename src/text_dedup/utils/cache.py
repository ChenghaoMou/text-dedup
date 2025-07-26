from pydantic import BaseModel

from text_dedup.utils.hashfunc import md5_digest


def config_fingerprint(model: BaseModel, suffix: str) -> str:
    return f"{md5_digest(model.model_dump_json().encode(), 'str')}_{suffix}"
