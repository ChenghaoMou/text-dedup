import pytest

from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha1_digest
from text_dedup.utils.hashfunc import sha1_hash
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
        for data, exp in zip(sample_data, expected, strict=True):
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

    def test_sha1_hash_32_bits(self, sample_data: list[bytes]) -> None:
        # Test 32-bit SHA1 hash using struct.unpack path
        for data in sample_data:
            result: int = sha1_hash(data, d=32)
            assert isinstance(result, int)
            assert 0 <= result < 2**32

    def test_sha1_hash_64_bits(self, sample_data: list[bytes]) -> None:
        # Test 64-bit SHA1 hash using struct.unpack path
        for data in sample_data:
            result: int = sha1_hash(data, d=64)
            assert isinstance(result, int)
            assert 0 <= result < 2**64

    def test_sha1_hash_arbitrary_bits(self) -> None:
        # Test arbitrary bit lengths using int.from_bytes path
        data: bytes = b"hello world"

        # Test various arbitrary bit lengths
        for bits in [8, 16, 24, 48, 80, 96, 120, 128, 144, 160]:
            result: int = sha1_hash(data, d=bits)
            assert isinstance(result, int)
            assert 0 <= result < 2**bits

    def test_sha1_hash_default_parameter(self, sample_data: list[bytes]) -> None:
        # Test that default d=32 works correctly
        for data in sample_data:
            result_default: int = sha1_hash(data)
            result_explicit: int = sha1_hash(data, d=32)
            assert result_default == result_explicit

    def test_sha1_hash_consistency(self) -> None:
        # Test that same input produces same output
        data: bytes = b"test consistency"

        # Test multiple calls with same parameters
        result1 = sha1_hash(data, d=32)
        result2 = sha1_hash(data, d=32)
        assert result1 == result2

        result3 = sha1_hash(data, d=64)
        result4 = sha1_hash(data, d=64)
        assert result3 == result4

        result5 = sha1_hash(data, d=128)
        result6 = sha1_hash(data, d=128)
        assert result5 == result6

    def test_sha1_hash_different_inputs(self) -> None:
        # Test that different inputs produce different outputs (with high probability)
        data1: bytes = b"hello world"
        data2: bytes = b"hello world!"

        # Test for different bit lengths
        for bits in [32, 64, 128]:
            result1: int = sha1_hash(data1, d=bits)
            result2: int = sha1_hash(data2, d=bits)
            # With high probability, different inputs should produce different hashes
            assert result1 != result2

    def test_sha1_hash_docstring_examples(self) -> None:
        # Test the examples from the docstring
        data: bytes = b"hello world"

        # These values come from the docstring examples
        assert sha1_hash(data, 32) == 896314922
        assert sha1_hash(data, 64) == 13028719972609469994
        assert sha1_hash(data, 128) == 310522945683037930239412421226792791594

    def test_sha1_hash_edge_cases(self) -> None:
        # Test edge cases

        # Empty data
        empty_data: bytes = b""
        result_empty = sha1_hash(empty_data, d=32)
        assert isinstance(result_empty, int)
        assert 0 <= result_empty < 2**32

        # Very small bit lengths
        data: bytes = b"test"
        result_8 = sha1_hash(data, d=8)
        assert 0 <= result_8 < 2**8

        # Large bit length
        result_160 = sha1_hash(data, d=160)  # Full SHA1 hash is 160 bits
        assert 0 <= result_160 < 2**160

    def test_sha1_hash_bit_length_relationship(self) -> None:
        # Test that smaller bit lengths are subsets of larger ones for struct paths
        data: bytes = b"test relationship"

        hash_32 = sha1_hash(data, d=32)
        hash_64 = sha1_hash(data, d=64)

        # For little-endian struct unpacking, the 32-bit hash should match
        # the lower 32 bits of the 64-bit hash
        assert hash_32 == (hash_64 & 0xFFFFFFFF)

    def test_sha1_hash_various_data_types(self) -> None:
        # Test with different types of byte data
        test_cases = [
            b"simple text",
            b"\x00\x01\x02\x03\x04",  # Binary data
            b"unicode: \xe4\xb8\xad\xe6\x96\x87",  # UTF-8 encoded text
            b"a" * 1000,  # Large repeated data
            b"".join(bytes([i]) for i in range(256)),  # All byte values
        ]

        for data in test_cases:
            # Test all three code paths
            result_32 = sha1_hash(data, d=32)
            result_64 = sha1_hash(data, d=64)
            result_arbitrary = sha1_hash(data, d=96)  # Arbitrary length

            assert isinstance(result_32, int)
            assert isinstance(result_64, int)
            assert isinstance(result_arbitrary, int)

            assert 0 <= result_32 < 2**32
            assert 0 <= result_64 < 2**64
            assert 0 <= result_arbitrary < 2**96
