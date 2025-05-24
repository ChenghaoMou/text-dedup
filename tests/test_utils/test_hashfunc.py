import pytest

from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha1_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_hash


@pytest.fixture
def sample_data() -> list[bytes]:
    return [b"hello world", b"python", b""]


class TestHashFunctions:
    def test_md5_digest_str(self, sample_data: list[bytes]) -> None:
        expected: list[str] = [
            "5eb63bbbe01eeed093cb22bb8f5acdc3",
            "23eeeb4347bdd26bfc6b7ee9a3b755dd",
            "d41d8cd98f00b204e9800998ecf8427e",
        ]
        for data, exp in zip(sample_data, expected):
            assert md5_digest(data, "str") == exp

    def test_md5_digest_bytes(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            digest: bytes = md5_digest(data, "bytes")
            assert isinstance(digest, bytes)
            assert len(digest) == 16
            assert md5_digest(data, "str") == digest.hex()

    def test_sha1_digest_str(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            digest_bytes: bytes = sha1_digest(data, "bytes")
            digest_str: str = sha1_digest(data, "str")
            assert digest_str == digest_bytes.hex()

    def test_sha1_digest_bytes(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            digest: bytes = sha1_digest(data, "bytes")
            assert isinstance(digest, bytes)
            assert len(digest) == 20

    def test_sha256_digest_str(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            digest_bytes: bytes = sha256_digest(data, "bytes")
            digest_str: str = sha256_digest(data, "str")
            assert digest_str == digest_bytes.hex()

    def test_sha256_digest_bytes(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            digest: bytes = sha256_digest(data, "bytes")
            assert isinstance(digest, bytes)
            assert len(digest) == 32

    def test_xxh3_digest_basic(self, sample_data: list[bytes]) -> None:
        for data in sample_data:
            result: int = xxh3_hash(data)
            assert isinstance(result, int)
            assert 0 <= result < 2**32

    def test_xxh3_digest_bits(self) -> None:
        data: bytes = b"hello world"

        hash_32: int = xxh3_hash(data, bits=32)
        hash_64: int = xxh3_hash(data, bits=64)
        hash_128: int = xxh3_hash(data, bits=128)

        assert 0 <= hash_32 < 2**32
        assert 0 <= hash_64 < 2**64
        assert 0 <= hash_128 < 2**128

        assert hash_32 == (hash_64 & 0xFFFFFFFF)
        assert hash_64 != (hash_128 & 0xFFFFFFFFFFFFFFFF)

    def test_xxh3_digest_seed(self) -> None:
        data: bytes = b"hello world"

        hash_1: int = xxh3_hash(data, seed=0)
        hash_2: int = xxh3_hash(data, seed=42)
        hash_3: int = xxh3_hash(data, seed=-42)

        assert hash_1 != hash_2
        assert hash_1 != hash_3
        assert hash_2 != hash_3

    def test_xxh3_digest_arbitrary_bits(self) -> None:
        data: bytes = b"hello world"

        hash_24: int = xxh3_hash(data, bits=24)
        hash_40: int = xxh3_hash(data, bits=40)

        assert 0 <= hash_24 < 2**24
        assert 0 <= hash_40 < 2**40


if __name__ == "__main__":
    pytest.main()
