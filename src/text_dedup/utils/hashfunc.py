import hashlib
import struct
from hashlib import md5
from hashlib import sha1
from hashlib import sha256
from typing import Literal
from typing import overload

import xxhash


@overload
def md5_digest(  # pyright: ignore[reportOverlappingOverload]
    data: bytes, return_type: Literal["str"] = "str"
) -> str:  # pragma: no cover
    ...


@overload
def md5_digest(data: bytes, return_type: Literal["bytes"] = "bytes") -> bytes:  # pragma: no cover
    ...


def md5_digest(data: bytes, return_type: Literal["str", "bytes"] = "str") -> bytes | str:
    h = md5(data, usedforsecurity=False)
    return h.hexdigest() if return_type == "str" else h.digest()


@overload
def sha1_digest(  # pyright: ignore[reportOverlappingOverload]
    data: bytes, return_type: Literal["str"] = "str"
) -> str:  # pragma: no cover
    ...


@overload
def sha1_digest(data: bytes, return_type: Literal["bytes"] = "bytes") -> bytes:  # pragma: no cover
    ...


def sha1_digest(data: bytes, return_type: Literal["str", "bytes"] = "str") -> bytes | str:
    h = sha1(data, usedforsecurity=False)
    return h.hexdigest() if return_type == "str" else h.digest()


@overload
def sha256_digest(  # pyright: ignore[reportOverlappingOverload]
    data: bytes, return_type: Literal["str"] = "str"
) -> str:  # pragma: no cover
    ...


@overload
def sha256_digest(data: bytes, return_type: Literal["bytes"] = "bytes") -> bytes:  # pragma: no cover
    ...


def sha256_digest(data: bytes, return_type: Literal["str", "bytes"] = "str") -> bytes | str:
    h = sha256(data, usedforsecurity=False)
    return h.hexdigest() if return_type == "str" else h.digest()


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    >>> sha1_hash(b"hello world", 128)
    310522945683037930239412421226792791594
    """
    if d == 32:
        return int(struct.unpack("<I", sha1(data, usedforsecurity=False).digest()[:4])[0])  # pyright: ignore[reportAny]
    if d == 64:
        return int(struct.unpack("<Q", sha1(data, usedforsecurity=False).digest()[:8])[0])  # pyright: ignore[reportAny]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(hashlib.sha1(data, usedforsecurity=False).digest()[: d // 8], byteorder="little")


def xxh3_hash(data: bytes, seed: int = 0, bits: int | Literal[32, 64, 128] = 32) -> int:
    match bits:
        case 32:
            return xxhash.xxh3_64_intdigest(data, seed) & 0xFFFFFFFF
        case 64:
            return xxhash.xxh3_64_intdigest(data, seed)
        case 128:
            return xxhash.xxh3_128_intdigest(data, seed)
        case _:
            return int.from_bytes(xxhash.xxh3_128_digest(data)[: bits // 8], byteorder="big")
