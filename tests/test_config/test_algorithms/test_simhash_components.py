import pytest
from bitarray import bitarray
from bitarray.util import int2ba

from text_dedup.config.algorithms.simhash import Mask
from text_dedup.config.algorithms.simhash import Permutation
from text_dedup.config.algorithms.simhash import _unsigned_hash
from text_dedup.config.algorithms.simhash import compute
from text_dedup.config.algorithms.simhash import hamming_distance
from text_dedup.utils.hashfunc import xxh3_hash


class TestMask:
    def test_mask_creation(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)

        assert mask.val == mask_val
        assert mask.width == 4
        assert mask.start == 2
        assert mask.end == 6

    def test_permute_positive_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)
        data = bitarray("0010100000")

        result = mask.permute(data, offset=2)

        expected = bitarray("1010000000")
        assert result == expected

    def test_permute_negative_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)
        data = bitarray("0010100000")

        result = mask.permute(data, offset=-2)

        expected = bitarray("0000101000")
        assert result == expected

    def test_permute_zero_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)
        data = bitarray("0010100000")

        result = mask.permute(data, offset=0)

        expected = bitarray("0010100000")
        assert result == expected

    def test_reverse_positive_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)
        data = bitarray("0010100000")

        result = mask.reverse(data, offset=2)

        expected = bitarray("0000101000")
        assert result == expected

    def test_reverse_negative_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)
        data = bitarray("0010100000")

        result = mask.reverse(data, offset=-2)

        expected = bitarray("1010000000")
        assert result == expected

    def test_reversed_positive_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)

        reversed_mask = mask.reversed(offset=2)

        assert reversed_mask.val == bitarray("1111000000")
        assert reversed_mask.width == 4
        assert reversed_mask.start == 2
        assert reversed_mask.end == 6

    def test_reversed_negative_offset(self) -> None:
        mask_val = bitarray("0011110000")
        mask = Mask(val=mask_val, width=4, start=2, end=6)

        reversed_mask = mask.reversed(offset=-2)

        assert reversed_mask.val == bitarray("0000111100")
        assert reversed_mask.width == 4
        assert reversed_mask.start == 2
        assert reversed_mask.end == 6


class TestPermutation:
    def test_permutation_creation(self) -> None:
        mask1 = Mask(val=bitarray("111000"), width=3, start=0, end=3)
        mask2 = Mask(val=bitarray("000111"), width=3, start=3, end=6)
        masks = [mask1, mask2]

        perm = Permutation(f=6, k=1, b=2, masks=masks)

        assert perm.f == 6
        assert perm.k == 1
        assert perm.b == 2
        assert len(perm.widths) == 2
        assert len(perm.offsets) == 2

    def test_permutation_invalid_widths(self) -> None:
        mask1 = Mask(val=bitarray("111000"), width=3, start=0, end=3)
        mask2 = Mask(val=bitarray("000110"), width=2, start=3, end=5)
        masks = [mask1, mask2]

        with pytest.raises(ValueError, match="sum of block widths"):
            Permutation(f=6, k=1, b=2, masks=masks)

    def test_permute_basic(self) -> None:
        mask1 = Mask(val=bitarray("1100"), width=2, start=0, end=2)
        mask2 = Mask(val=bitarray("0011"), width=2, start=2, end=4)
        masks = [mask1, mask2]

        perm = Permutation(f=4, k=1, b=2, masks=masks)
        data = bitarray("1010")

        result = perm.permute(data)

        assert len(result) == 4
        assert isinstance(result, bitarray)

    def test_reverse_basic(self) -> None:
        mask1 = Mask(val=bitarray("1100"), width=2, start=0, end=2)
        mask2 = Mask(val=bitarray("0011"), width=2, start=2, end=4)
        masks = [mask1, mask2]

        perm = Permutation(f=4, k=1, b=2, masks=masks)
        data = bitarray("1010")

        permuted = perm.permute(data)
        reversed_data = perm.reverse(permuted)

        assert reversed_data == data

    def test_permute_reverse_roundtrip(self) -> None:
        mask1 = Mask(val=bitarray("11110000"), width=4, start=0, end=4)
        mask2 = Mask(val=bitarray("00001111"), width=4, start=4, end=8)
        masks = [mask1, mask2]

        perm = Permutation(f=8, k=1, b=2, masks=masks)

        test_cases = [
            bitarray("10101010"),
            bitarray("11110000"),
            bitarray("00001111"),
            bitarray("11111111"),
            bitarray("00000000"),
        ]

        for data in test_cases:
            permuted = perm.permute(data)
            reversed_data = perm.reverse(permuted)
            assert reversed_data == data

    def test_search_mask_creation(self) -> None:
        mask1 = Mask(val=bitarray("111000"), width=3, start=0, end=3)
        mask2 = Mask(val=bitarray("000111"), width=3, start=3, end=6)
        masks = [mask1, mask2]

        perm = Permutation(f=6, k=1, b=2, masks=masks)

        assert len(perm.search_mask) == 6
        expected_mask = bitarray("111000")
        assert perm.search_mask == expected_mask


class TestHammingDistance:
    def test_identical_arrays(self) -> None:
        a = bitarray("1010")
        b = bitarray("1010")
        assert hamming_distance(a, b) == 0

    def test_completely_different(self) -> None:
        a = bitarray("1111")
        b = bitarray("0000")
        assert hamming_distance(a, b) == 4

    def test_single_bit_difference(self) -> None:
        a = bitarray("1010")
        b = bitarray("0010")
        assert hamming_distance(a, b) == 1

    def test_multiple_bit_differences(self) -> None:
        a = bitarray("10101010")
        b = bitarray("01010101")
        assert hamming_distance(a, b) == 8

    def test_empty_arrays(self) -> None:
        a = bitarray("")
        b = bitarray("")
        assert hamming_distance(a, b) == 0

    def test_various_distances(self) -> None:
        test_cases = [
            (bitarray("1100"), bitarray("1100"), 0),
            (bitarray("1100"), bitarray("1101"), 1),
            (bitarray("1100"), bitarray("0011"), 4),
            (bitarray("11110000"), bitarray("10101010"), 4),
        ]

        for a, b, expected in test_cases:
            assert hamming_distance(a, b) == expected


class TestUnsignedHash:
    def test_hash_length(self) -> None:
        data = b"hello world"
        result = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=64), length=8)

        assert len(result) == 64

    def test_hash_consistency(self) -> None:
        data = b"test data"
        result1 = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=64), length=8)
        result2 = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=64), length=8)

        assert result1 == result2

    def test_different_inputs_different_hashes(self) -> None:
        data1 = b"hello"
        data2 = b"world"

        result1 = _unsigned_hash(data1, lambda x: xxh3_hash(x, bits=64), length=8)
        result2 = _unsigned_hash(data2, lambda x: xxh3_hash(x, bits=64), length=8)

        assert result1 != result2

    def test_empty_input(self) -> None:
        data = b""
        result = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=64), length=8)

        assert len(result) == 64
        assert isinstance(result, bitarray)

    def test_different_lengths(self) -> None:
        data = b"test"

        result_32 = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=32), length=4)
        result_64 = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=64), length=8)
        result_128 = _unsigned_hash(data, lambda x: xxh3_hash(x, bits=128), length=16)

        assert len(result_32) == 32
        assert len(result_64) == 64
        assert len(result_128) == 128


class TestCompute:
    def test_compute_single_hash(self) -> None:
        hash1 = int2ba(13352372148217134600, length=64)
        result = compute([hash1])

        assert len(result) == 64
        assert isinstance(result, bitarray)

    def test_compute_multiple_hashes(self) -> None:
        hash1 = int2ba(13352372148217134600, length=64)
        hash2 = int2ba(5020219685658847592, length=64)

        result = compute([hash1, hash2])

        assert len(result) == 64
        assert isinstance(result, bitarray)

    def test_compute_empty_list(self) -> None:
        with pytest.raises(ValueError, match="Cannot compute simhash from empty hash list"):
            compute([])

    def test_compute_consistency(self) -> None:
        hashes = [int2ba(i, length=32) for i in [100, 200, 300]]

        result1 = compute(hashes)
        result2 = compute(hashes)

        assert result1 == result2

    def test_compute_all_zeros(self) -> None:
        hashes = [int2ba(0, length=64) for _ in range(5)]
        result = compute(hashes)

        assert len(result) == 64
        assert result == bitarray("0" * 64)

    def test_compute_all_ones(self) -> None:
        hashes = [int2ba((1 << 64) - 1, length=64) for _ in range(5)]
        result = compute(hashes)

        assert len(result) == 64
        assert result == bitarray("1" * 64)

    def test_compute_mixed_values(self) -> None:
        hashes = [
            int2ba(0b1010, length=8),
            int2ba(0b1100, length=8),
            int2ba(0b0011, length=8),
        ]

        result = compute(hashes)

        assert len(result) == 8
        assert isinstance(result, bitarray)

    def test_compute_majority_voting(self) -> None:
        hashes = [
            bitarray("1111"),
            bitarray("1110"),
            bitarray("1100"),
        ]

        result = compute(hashes)

        expected = bitarray("1110")
        assert result == expected

    def test_compute_large_number_of_hashes(self) -> None:
        hashes = [int2ba(i * 12345, length=64) for i in range(100)]

        result = compute(hashes)

        assert len(result) == 64
        assert isinstance(result, bitarray)

    def test_compute_different_lengths(self) -> None:
        hashes_32 = [int2ba(i, length=32) for i in [100, 200, 300]]
        result_32 = compute(hashes_32)
        assert len(result_32) == 32

        hashes_128 = [int2ba(i, length=128) for i in [100, 200, 300]]
        result_128 = compute(hashes_128)
        assert len(result_128) == 128

    def test_compute_tie_breaking(self) -> None:
        hashes = [
            bitarray("10"),
            bitarray("01"),
        ]

        result = compute(hashes)

        assert len(result) == 2
        assert result == bitarray("00")
