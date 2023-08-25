import hashlib
import struct
from hashlib import md5
from hashlib import sha256

import xxhash
from xxhash import xxh3_64
from xxhash import xxh3_64_digest
from xxhash import xxh3_128
from xxhash import xxh3_128_digest


def md5_digest(data: bytes) -> bytes:
    """
    Generate a md5 hash in bytestring form from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.

    Returns
    -------
    bytes
        The hash value in raw byte strings.

    Examples
    --------
    # raw byte strings cause problems on doctests
    >>> int.from_bytes(md5_digest(b"hello world"),"little")
    260265716838465564751810390803223393886
    >>> len(md5_digest(b"hello world"))
    16
    """
    return md5(data).digest()


def md5_hexdigest(data: bytes) -> str:
    """
    Generate a md5 hex hash from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.

    Returns
    -------
    str
        The hex hash value.

    Examples
    --------
    >>> md5_hexdigest(b"hello world")
    '5eb63bbbe01eeed093cb22bb8f5acdc3'
    >>> len(md5_hexdigest(b"hello world"))
    32
    """
    return md5(data).hexdigest()


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
    digest = hashlib.sha1(data).digest()
    match d:
        case 16:
            return struct.unpack("<H", digest[:2])[0]
        case 32:
            return struct.unpack("<I", digest[:4])[0]
        case 64:
            return struct.unpack("<Q", digest[:8])[0]
        # struct is faster but does not support arbitrary bit lengths
        case _:
            return int.from_bytes(digest[: d // 8], byteorder="little")


def sha256_digest(data: bytes) -> bytes:
    """
    Generate a sha256 hash in bytestring form from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.

    Returns
    -------
    bytes
        The hash value in raw byte strings.

    Examples
    --------
    # raw byte strings cause problems on doctests
    >>> int.from_bytes(sha256_digest(b"hello world"),"little")
    105752752996721010526070019734402373604975086831773275823333741804099920678329
    >>> len(sha256_digest(b"hello world"))
    32
    """
    return sha256(data).digest()


def sha256_hexdigest(data: bytes) -> str:
    """
    Generate a sha256 hex hash from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.

    Returns
    -------
    str
        The hex hash value.

    Examples
    --------
    >>> sha256_hexdigest(b"hello world")
    'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    >>> len(sha256_hexdigest(b"hello world"))
    64
    """
    return sha256(data).hexdigest()


def xxh3_16hash(data: bytes, seed: int = 0) -> int:
    """
    Generate a 16-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.
    This is useful as a special purpose xxhash when you only want 16 bits.
    bit masked xxh3_64 hashes are faster than xxh32 in modern systems.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    seed : int
        xxhashes can all be seeded. Default is int=0

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> xxh3_16hash(b"hello world")
    39051
    >>> xxh3_16hash(b"hello world",seed=42)
    13198
    >>> xxh3_16hash(b"hello world",seed=-42)
    34281
    """
    return xxhash.xxh3_64_intdigest(data, seed) & 0xFFFF


def xxh3_32hash(data: bytes, seed: int = 0) -> int:
    """
    Generate a 32-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.
    This is useful as a special purpose xxhash when you only want 32bits.
    bit masked xxh3_64 hashes are faster than xxh32 in modern systems.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    seed : int
        xxhashes can all be seeded. Default is int=0

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> xxh3_32hash(b"hello world")
    1088854155
    >>> xxh3_32hash(b"hello world",seed=42)
    3913102222
    >>> xxh3_32hash(b"hello world",seed=-42)
    3721037289
    """
    return xxhash.xxh3_64_intdigest(data, seed) & 0xFFFFFFFF


def xxh3_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.
    This is useful as a general purpose xxhash that can take multiple `d` values

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
    >>> xxh3_hash(b"hello world", 16)
    39051
    >>> xxh3_hash(b"hello world", 32)
    1088854155
    >>> xxh3_hash(b"hello world", 64)
    15296390279056496779
    >>> xxh3_hash(b"hello world", 128)
    297150157938599054391163723952090887879
    """
    BITMASKS = {16: 0xFFFF, 32: 0xFFFFFFFF, 64: 0xFFFFFFFFFFFFFFFF}
    match d:
        case d_val if d_val in BITMASKS:
            return xxhash.xxh3_64_intdigest(data) & BITMASKS[d_val]
        case 128:
            return xxhash.xxh3_128_intdigest(data)
    # fall back
    return int.from_bytes(xxhash.xxh3_128_digest(data)[: d // 8], byteorder="big")


__all__ = [
    "md5",
    "sha256",
    "sha1_hash",
    "xxh3_64",
    "xxh3_64_digest",
    "xxh3_128",
    "xxh3_128_digest",
    "xxh3_hash",
    "xxh3_16hash",
    "xxh3_32hash",
]
