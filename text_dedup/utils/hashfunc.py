import hashlib
import struct
import xxhash

from hashlib import md5, sha256
from blake3 import blake3

from xxhash import xxh3_128, xxh3_64_digest, xxh3_128_digest


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
        return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]
    if d == 64:
        return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(hashlib.sha1(data).digest()[: d // 8], byteorder="little")


def xxh3_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.
        According to this value, chooses empirically found best xxh3 hasher.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> xxh3_hash(b"hello world", 32)
    1088854155
    >>> xxh3_hash(b"hello world", 64)
    15296390279056496779
    >>> xxh3_hash(b"hello world", 128)
    297150157938599054391163723952090887879
    """
    match d:
        case 32:
            # with sse2 or later, xxh3 is much faster
            # with avx, the difference is much larger
            return xxhash.xxh3_64_intdigest(data) & 0xFFFFFFFF
        case 64:
            return xxhash.xxh3_64_intdigest(data)
        case 128:
            return xxhash.xxh3_128_intdigest(data)
    # fall back
    return int.from_bytes(xxhash.xxh3_128_digest(data)[: d // 8], byteorder="big")
