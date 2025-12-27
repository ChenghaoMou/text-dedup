import pytest

from text_dedup.utils.jaccard import cluster_jaccard_similarity
from text_dedup.utils.jaccard import jaccard_similarity


class TestJaccardSimilarity:
    @pytest.mark.parametrize(
        "doc1, doc2, expected, description",
        [  # pyright: ignore[reportUnknownArgumentType]
            # Identical sets
            ({"hello", "world", "test"}, {"hello", "world", "test"}, 1.0, "identical_sets"),
            # Empty sets (both empty sets are identical, so similarity is 1.0)
            (set(), set(), 1.0, "empty_sets"),
            # One empty set
            ({"hello", "world"}, set(), 0.0, "one_empty_set"),
            # No overlap
            ({"hello", "world"}, {"foo", "bar"}, 0.0, "no_overlap"),
            # Partial overlap
            ({"hello", "world", "test"}, {"hello", "foo", "bar"}, 1.0 / 5.0, "partial_overlap"),
            # Subset relationship
            ({"hello", "world"}, {"hello", "world", "test", "foo"}, 2.0 / 4.0, "subset_relationship"),
            # Bytes sets
            ({b"hello", b"world", b"test"}, {b"hello", b"foo", b"bar"}, 1.0 / 5.0, "with_bytes"),
            # Mixed sizes
            ({"a"}, {"a", "b", "c", "d", "e"}, 1.0 / 5.0, "mixed_sizes"),
            # Single element identical
            ({"hello"}, {"hello"}, 1.0, "single_element_identical"),
            # Single element no overlap
            ({"hello"}, {"world"}, 0.0, "single_element_no_overlap"),
        ],
    )
    def test_jaccard_similarity_cases(self, doc1: set, doc2: set, expected: float, description: str) -> None:
        result = jaccard_similarity(doc1, doc2)
        assert result == expected

    def test_jaccard_similarity_symmetry(self) -> None:
        doc1 = {"hello", "world", "test"}
        doc2 = {"hello", "foo", "bar", "baz"}

        result1 = jaccard_similarity(doc1, doc2)
        result2 = jaccard_similarity(doc2, doc1)
        assert result1 == result2

    def test_jaccard_similarity_one_empty_set_reverse(self) -> None:
        doc1 = {"hello", "world"}
        doc2: set[str] = set()

        result = jaccard_similarity(doc1, doc2)
        result_reverse = jaccard_similarity(doc2, doc1)
        assert result == result_reverse == 0.0


class TestClusterJaccardSimilarity:
    @pytest.mark.parametrize(
        "cluster, threshold, expected_similarities, expected_fp_rate, description",
        [
            # Empty cluster
            ([], 0.5, [], 0, "empty_cluster"),
            # Single document
            ([{b"hello", b"world"}], 0.5, [], 0, "single_document"),
            # Two identical documents
            (
                [{b"hello", b"world", b"test"}, {b"hello", b"world", b"test"}],
                0.5,
                [1.0, 1.0],
                0.0,
                "two_identical_documents",
            ),
            # No overlap documents
            ([{b"hello", b"world"}, {b"foo", b"bar"}], 0.5, [0.0, 0.0], 1.0, "no_overlap_documents"),
        ],
    )
    def test_cluster_jaccard_similarity_basic_cases(
        self,
        cluster: list[set[bytes]],
        threshold: float,
        expected_similarities: list[float],
        expected_fp_rate: float,
        description: str,
    ) -> None:
        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert similarities == expected_similarities
        assert fp_rate == expected_fp_rate

    def test_cluster_jaccard_similarity_partial_overlap(self) -> None:
        doc1 = {b"hello", b"world", b"test"}
        doc2 = {b"hello", b"foo", b"bar"}
        doc3 = {b"baz", b"qux"}
        cluster = [doc1, doc2, doc3]
        threshold = 0.3

        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert len(similarities) == 3

        # doc1's best match is with doc2: intersection {b"hello"}, union size 5, similarity = 0.2
        assert similarities[0] == 0.2

        # doc2's best match is with doc1: intersection {b"hello"}, union size 5, similarity = 0.2
        assert similarities[1] == 0.2

        # doc3's best match is with either doc1 or doc2: no overlap, similarity = 0.0
        assert similarities[2] == 0.0

        # All three documents have similarities < 0.3, so fp_rate = 3/3 = 1.0
        assert fp_rate == 1.0

    def test_cluster_jaccard_similarity_mixed_threshold_results(self) -> None:
        doc1 = {b"a", b"b", b"c", b"d", b"e"}  # 5 elements
        doc2 = {b"a", b"b", b"c"}  # 3 elements, 3/5 = 0.6 similarity with doc1
        doc3 = {b"x", b"y", b"z"}  # 3 elements, 0 similarity with others
        cluster = [doc1, doc2, doc3]
        threshold = 0.5

        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert len(similarities) == 3
        assert similarities[0] == 0.6  # doc1's best match with doc2
        assert similarities[1] == 0.6  # doc2's best match with doc1
        assert similarities[2] == 0.0  # doc3 has no overlap with others

        # Only doc3 has similarity < 0.5, so fp_rate = 1/3
        assert fp_rate == pytest.approx(1.0 / 3.0)

    def test_cluster_jaccard_similarity_all_above_threshold(self) -> None:
        doc1 = {b"a", b"b"}
        doc2 = {b"a", b"c"}  # similarity with doc1 = 1/3 ≈ 0.33
        doc3 = {b"a", b"d"}  # similarity with doc1 = 1/3 ≈ 0.33, with doc2 = 1/4 = 0.25
        cluster = [doc1, doc2, doc3]
        threshold = 0.2

        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert len(similarities) == 3
        assert similarities[0] == pytest.approx(1.0 / 3.0)  # doc1's best match
        assert similarities[1] == pytest.approx(1.0 / 3.0)  # doc2's best match
        assert similarities[2] == pytest.approx(1.0 / 3.0)  # doc3's best match with doc1

        # All similarities >= 0.2, so fp_rate = 0
        assert fp_rate == 0.0

    def test_cluster_jaccard_similarity_large_cluster(self) -> None:
        # Create cluster where first document overlaps with all others
        base_doc = {b"common", b"base"}
        cluster = [base_doc]

        # Add documents that each share "common" with base_doc
        for i in range(5):
            doc = {b"common", f"unique_{i}".encode()}
            cluster.append(doc)

        threshold = 0.4
        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert len(similarities) == 6

        # Each document should have similarity 1/3 ≈ 0.33 with base_doc or other docs
        for similarity in similarities:
            assert similarity == pytest.approx(1.0 / 3.0, abs=0.01)

        # All similarities are below 0.4, so fp_rate = 1.0
        assert fp_rate == 1.0

    @pytest.mark.parametrize(
        "cluster, threshold, expected_similarities, expected_fp_rate, description",
        [
            # All empty sets (empty sets are identical, so similarity is 1.0)
            ([set(), set(), set()], 0.5, [1.0, 1.0, 1.0], 0.0, "all_empty_sets"),
            # One empty set among non-overlapping docs
            ([{b"hello", b"world"}, set(), {b"foo", b"bar"}], 0.5, [0.0, 0.0, 0.0], 1.0, "one_empty_set"),
        ],
    )
    def test_cluster_jaccard_similarity_edge_cases(
        self,
        cluster: list[set[bytes]],
        threshold: float,
        expected_similarities: list[float],
        expected_fp_rate: float,
        description: str,
    ) -> None:
        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert len(similarities) == len(cluster)
        assert similarities == expected_similarities
        assert fp_rate == expected_fp_rate

    def test_cluster_jaccard_similarity_threshold_boundary(self) -> None:
        doc1 = {b"a", b"b"}
        doc2 = {b"a", b"c"}  # similarity = 1/3 ≈ 0.3333
        cluster = [doc1, doc2]

        # Test with threshold just below the similarity
        threshold = 0.33
        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)
        assert fp_rate == 0.0  # Both docs above threshold

        # Test with threshold just above the similarity
        threshold = 0.34
        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)
        assert fp_rate == 1.0  # Both docs below threshold

    def test_cluster_jaccard_similarity_value_error_handling(self) -> None:
        # This test ensures the ValueError handling in the max() call works correctly
        # In practice, this should rarely happen since we filter out the same document
        doc1 = {b"unique"}
        cluster = [doc1]

        # Add the same document reference again to force an edge case
        cluster.append(doc1)
        threshold = 0.5

        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        # Both positions should have similarity 1.0 since they're identical
        assert len(similarities) == 2
        assert similarities[0] == 1.0
        assert similarities[1] == 1.0
        assert fp_rate == 0.0

    def test_cluster_jaccard_similarity_type_consistency(self) -> None:
        # Test that the function works with bytes sets as expected
        doc1 = {b"hello", b"world"}
        doc2 = {b"hello", b"test"}
        cluster = [doc1, doc2]
        threshold = 0.5

        similarities, fp_rate = cluster_jaccard_similarity(cluster, threshold)

        assert isinstance(similarities, list)
        assert isinstance(fp_rate, float)
        assert all(isinstance(sim, float) for sim in similarities)
        assert 0.0 <= fp_rate <= 1.0
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
