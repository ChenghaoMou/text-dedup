from collections.abc import Iterator

import pytest

from text_dedup.utils.tokenization import ngrams


class TestNgrams:
    @pytest.mark.parametrize(
        "sequence, n, min_length, expected, description",
        [
            # Basic trigram
            (
                ["hello", "world", "this", "is", "test"],
                3,
                5,
                [("hello", "world", "this"), ("world", "this", "is"), ("this", "is", "test")],
                "basic_trigram",
            ),
            # Basic bigram
            (
                ["hello", "world", "test"],
                2,
                3,
                [("hello", "world"), ("world", "test")],
                "basic_bigram",
            ),
            # Single token
            (
                ["hello", "world", "test"],
                1,
                3,
                [("hello",), ("world",), ("test",)],
                "single_token",
            ),
            # Sequence equals n
            (
                ["hello", "world", "test"],
                3,
                3,
                [("hello", "world", "test")],
                "sequence_equals_n",
            ),
        ],
    )
    def test_ngrams_basic_cases(
        self,
        sequence: list[str],
        n: int,
        min_length: int,
        expected: list[tuple[str, ...]],
        description: str,
    ) -> None:
        result = list(ngrams(sequence, n, min_length))
        assert result == expected

    @pytest.mark.parametrize(
        "sequence, n, min_length, expected, description",
        [
            # Sequence shorter than min_length
            (["hello", "world"], 2, 5, [], "sequence_shorter_than_min_length"),
            # Sequence equals min_length
            (
                ["hello", "world", "test", "case", "example"],
                3,
                5,
                [("hello", "world", "test"), ("world", "test", "case"), ("test", "case", "example")],
                "sequence_equals_min_length",
            ),
            # Sequence longer than n but shorter than min_length
            (["hello", "world", "test"], 2, 5, [], "sequence_longer_than_n_shorter_than_min_length"),
            # Sequence shorter than n but longer than min_length
            (
                ["hello", "world", "test", "case", "example", "text"],
                8,
                5,
                [("hello", "world", "test", "case", "example", "text")],
                "sequence_shorter_than_n_longer_than_min_length",
            ),
        ],
    )
    def test_ngrams_length_relationships(
        self,
        sequence: list[str],
        n: int,
        min_length: int,
        expected: list[tuple[str, ...]],
        description: str,
    ) -> None:
        result = list(ngrams(sequence, n, min_length))
        assert result == expected

    @pytest.mark.parametrize(
        "sequence, n, min_length, expected, description",
        [
            # Default min_length (implicit 5)
            (["a", "b", "c", "d"], 2, None, [], "default_min_length"),
            # Empty sequence
            ([], 2, None, [], "empty_sequence"),
            # Single word below min_length
            (["hello"], 1, None, [], "single_word_below_min_length"),
            # Single word with low min_length
            (["hello"], 1, 1, [("hello",)], "single_word_low_min_length"),
            # Zero n
            (["hello", "world", "test", "case", "example"], 0, None, [], "zero_n"),
            # Edge case: n equals sequence length equals min_length
            (["a", "b", "c", "d", "e"], 5, 5, [("a", "b", "c", "d", "e")], "n_equals_sequence_equals_min_length"),
        ],
    )
    def test_ngrams_edge_cases(
        self,
        sequence: list[str],
        n: int,
        min_length: int | None,
        expected: list[tuple[str, ...]],
        description: str,
    ) -> None:
        result = list(ngrams(sequence, n)) if min_length is None else list(ngrams(sequence, n, min_length))
        assert result == expected, f"{description}: {result} != {expected}"

    def test_ngrams_large_n(self) -> None:
        sequence = ["a", "b", "c", "d", "e", "f", "g"]
        n = 5
        min_length = 6

        result = list(ngrams(sequence, n, min_length))

        expected = [
            ("a", "b", "c", "d", "e"),
            ("b", "c", "d", "e", "f"),
            ("c", "d", "e", "f", "g"),
        ]
        assert result == expected

    def test_ngrams_returns_iterator(self) -> None:
        sequence = ["hello", "world", "test", "case", "example"]
        n = 2

        result = ngrams(sequence, n)

        # Should return an iterator
        assert isinstance(result, Iterator)

        # Should be able to iterate through it
        first_ngram = next(result)
        assert first_ngram == ("hello", "world")

    def test_ngrams_iterator_can_be_consumed_multiple_times(self) -> None:
        sequence = ["hello", "world", "test", "case", "example"]
        n = 2

        # Create iterator and convert to list twice
        result1 = list(ngrams(sequence, n))
        result2 = list(ngrams(sequence, n))

        # Both should give the same results
        expected = [
            ("hello", "world"),
            ("world", "test"),
            ("test", "case"),
            ("case", "example"),
        ]
        assert result1 == expected
        assert result2 == expected

    @pytest.mark.parametrize(
        "sequence, n, expected, description",
        [
            # Repeated elements
            (
                ["hello", "hello", "world", "hello", "test"],
                3,
                [("hello", "hello", "world"), ("hello", "world", "hello"), ("world", "hello", "test")],
                "repeated_elements",
            ),
            # Special characters
            (
                ["hello", "world!", "@test", "#case", "$example"],
                2,
                [("hello", "world!"), ("world!", "@test"), ("@test", "#case"), ("#case", "$example")],
                "special_characters",
            ),
        ],
    )
    def test_ngrams_content_variations(
        self,
        sequence: list[str],
        n: int,
        expected: list[tuple[str, ...]],
        description: str,
    ) -> None:
        result = list(ngrams(sequence, n))
        assert result == expected

    def test_ngrams_long_sequence(self) -> None:
        # Test with longer sequence to ensure performance is reasonable
        sequence = [f"word_{i}" for i in range(100)]
        n = 4
        min_length = 50

        result = list(ngrams(sequence, n, min_length))

        # Should have 100 - 4 + 1 = 97 ngrams
        assert len(result) == 97

        # Check first and last ngrams
        assert result[0] == ("word_0", "word_1", "word_2", "word_3")
        assert result[-1] == ("word_96", "word_97", "word_98", "word_99")

    def test_ngrams_edge_case_n_equals_sequence_length_equals_min_length(self) -> None:
        sequence = ["a", "b", "c", "d", "e"]
        n = 5
        min_length = 5

        result = list(ngrams(sequence, n, min_length))

        # Should return single tuple with entire sequence
        expected = [("a", "b", "c", "d", "e")]
        assert result == expected

    def test_ngrams_zero_n(self) -> None:
        sequence = ["hello", "world", "test", "case", "example"]
        n = 0

        # This might be an edge case - test actual behavior
        result = list(ngrams(sequence, n))

        # With n=0, tee() creates empty iterables, resulting in empty zip
        # This tests the actual implementation behavior
        expected: list[tuple[str, ...]] = []
        assert result == expected

    def test_ngrams_negative_n(self) -> None:
        sequence = ["hello", "world", "test", "case", "example"]
        n = -1

        # Test behavior with negative n - this should raise ValueError
        with pytest.raises(ValueError, match="n must be >= 0"):
            list(ngrams(sequence, n))

    @pytest.mark.parametrize(
        "sequence, n, min_length, expected, description",
        [
            # Min length larger than sequence
            (["hello", "world", "test"], 2, 10, [], "min_length_larger_than_sequence"),
        ],
    )
    def test_ngrams_boundary_conditions(
        self,
        sequence: list[str],
        n: int,
        min_length: int,
        expected: list[tuple[str, ...]],
        description: str,
    ) -> None:
        result = list(ngrams(sequence, n, min_length))
        assert result == expected

    def test_ngrams_tuple_immutability(self) -> None:
        sequence = ["hello", "world", "test", "case", "example"]
        n = 2

        result = list(ngrams(sequence, n))

        # Each result should be a tuple (immutable)
        for ngram in result:
            assert isinstance(ngram, tuple)

        # Test that tuples contain the expected strings
        assert all(isinstance(token, str) for ngram in result for token in ngram)

    def test_ngrams_memory_efficiency_with_iterator(self) -> None:
        # Test that the function returns an iterator, not a list
        sequence = ["hello", "world", "test", "case", "example"]
        n = 2

        result = ngrams(sequence, n)

        # Should be an iterator, not a materialized list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

        # Should be able to get elements one by one
        first = next(result)
        second = next(result)

        assert first == ("hello", "world")
        assert second == ("world", "test")

    def test_ngrams_type_annotations(self) -> None:
        # Test that function works with proper type annotations
        sequence = ["hello", "world", "test", "case", "example"]
        n = 3

        result: Iterator[tuple[str, ...]] = ngrams(sequence, n)
        ngram_list: list[tuple[str, ...]] = list(result)

        assert len(ngram_list) == 3
        assert all(len(ngram) == 3 for ngram in ngram_list)
