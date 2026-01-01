import pytest

from benchmarks.utils import classify_prediction
from benchmarks.utils import clusters_to_predictions_minhash
from benchmarks.utils import clusters_to_predictions_simhash
from benchmarks.utils import f1_score
from benchmarks.utils import precision
from benchmarks.utils import recall


class TestPrecision:
    def test_perfect_precision(self) -> None:
        duplicates = {"a", "b", "c"}
        predictions = {"a", "b", "c"}
        assert precision(duplicates, predictions) == 1.0

    def test_partial_precision(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"a", "b", "c", "d"}
        assert precision(duplicates, predictions) == 0.5

    def test_zero_precision(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"c", "d"}
        assert precision(duplicates, predictions) == 0.0

    def test_empty_predictions(self) -> None:
        duplicates = {"a", "b"}
        predictions: set = set()
        assert precision(duplicates, predictions) == 0.0

    def test_empty_duplicates(self) -> None:
        duplicates: set = set()
        predictions = {"a", "b"}
        assert precision(duplicates, predictions) == 0.0

    def test_both_empty(self) -> None:
        duplicates: set = set()
        predictions: set = set()
        assert precision(duplicates, predictions) == 0.0

    def test_single_element_match(self) -> None:
        duplicates = {"a"}
        predictions = {"a"}
        assert precision(duplicates, predictions) == 1.0

    def test_single_element_no_match(self) -> None:
        duplicates = {"a"}
        predictions = {"b"}
        assert precision(duplicates, predictions) == 0.0

    def test_subset_predictions(self) -> None:
        duplicates = {"a", "b", "c", "d"}
        predictions = {"a", "b"}
        assert precision(duplicates, predictions) == 1.0

    def test_overlapping_sets(self) -> None:
        duplicates = {"a", "b", "c"}
        predictions = {"b", "c", "d", "e"}
        assert precision(duplicates, predictions) == 0.5


class TestRecall:
    def test_perfect_recall(self) -> None:
        duplicates = {"a", "b", "c"}
        predictions = {"a", "b", "c"}
        assert recall(duplicates, predictions) == 1.0

    def test_partial_recall(self) -> None:
        duplicates = {"a", "b", "c", "d"}
        predictions = {"a", "b"}
        assert recall(duplicates, predictions) == 0.5

    def test_zero_recall(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"c", "d"}
        assert recall(duplicates, predictions) == 0.0

    def test_empty_duplicates(self) -> None:
        duplicates: set = set()
        predictions = {"a", "b"}
        assert recall(duplicates, predictions) == 1.0

    def test_empty_predictions(self) -> None:
        duplicates = {"a", "b"}
        predictions: set = set()
        assert recall(duplicates, predictions) == 0.0

    def test_both_empty(self) -> None:
        duplicates: set = set()
        predictions: set = set()
        assert recall(duplicates, predictions) == 1.0

    def test_single_element_match(self) -> None:
        duplicates = {"a"}
        predictions = {"a"}
        assert recall(duplicates, predictions) == 1.0

    def test_single_element_no_match(self) -> None:
        duplicates = {"a"}
        predictions = {"b"}
        assert recall(duplicates, predictions) == 0.0

    def test_superset_predictions(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"a", "b", "c", "d"}
        assert recall(duplicates, predictions) == 1.0

    def test_overlapping_sets(self) -> None:
        duplicates = {"a", "b", "c", "d"}
        predictions = {"b", "c"}
        assert recall(duplicates, predictions) == 0.5


class TestF1Score:
    def test_perfect_f1(self) -> None:
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero_f1(self) -> None:
        assert f1_score(0.0, 0.0) == 0.0

    def test_balanced_scores(self) -> None:
        result = f1_score(0.5, 0.5)
        assert result == 0.5

    def test_high_precision_low_recall(self) -> None:
        result = f1_score(0.9, 0.3)
        assert result == pytest.approx(0.45, abs=0.01)

    def test_low_precision_high_recall(self) -> None:
        result = f1_score(0.3, 0.9)
        assert result == pytest.approx(0.45, abs=0.01)

    def test_harmonic_mean_property(self) -> None:
        p, r = 0.6, 0.8
        expected = 2 * p * r / (p + r)
        assert f1_score(p, r) == pytest.approx(expected)

    def test_zero_precision(self) -> None:
        assert f1_score(0.0, 0.5) == 0.0

    def test_zero_recall(self) -> None:
        assert f1_score(0.5, 0.0) == 0.0

    def test_one_precision_zero_recall(self) -> None:
        assert f1_score(1.0, 0.0) == 0.0

    def test_various_combinations(self) -> None:
        test_cases = [
            (0.8, 0.6, 2 * 0.8 * 0.6 / (0.8 + 0.6)),
            (0.75, 0.75, 0.75),
            (0.1, 0.9, 2 * 0.1 * 0.9 / (0.1 + 0.9)),
        ]
        for p, r, expected in test_cases:
            assert f1_score(p, r) == pytest.approx(expected, abs=0.001)


class TestClassifyPrediction:
    def test_true_positive(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"a", "b", "c"}
        assert classify_prediction(duplicates, predictions) == "TP"

    def test_true_positive_exact(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"a", "b"}
        assert classify_prediction(duplicates, predictions) == "TP"

    def test_true_negative(self) -> None:
        duplicates: set = set()
        predictions: set = set()
        assert classify_prediction(duplicates, predictions) == "TN"

    def test_false_positive_no_duplicates(self) -> None:
        duplicates: set = set()
        predictions = {"a", "b"}
        assert classify_prediction(duplicates, predictions) == "FP"

    def test_false_positive_partial_match(self) -> None:
        duplicates = {"a"}
        predictions = {"a", "b"}
        assert classify_prediction(duplicates, predictions) == "TP"

    def test_false_positive_no_overlap(self) -> None:
        duplicates = {"a", "b"}
        predictions = {"c", "d"}
        assert classify_prediction(duplicates, predictions) == "FP"

    def test_false_negative(self) -> None:
        duplicates = {"a", "b"}
        predictions: set = set()
        assert classify_prediction(duplicates, predictions) == "FN"

    def test_single_duplicate_found(self) -> None:
        duplicates = {"a"}
        predictions = {"a"}
        assert classify_prediction(duplicates, predictions) == "TP"

    def test_single_duplicate_not_found(self) -> None:
        duplicates = {"a"}
        predictions: set = set()
        assert classify_prediction(duplicates, predictions) == "FN"

    def test_subset_relationship(self) -> None:
        duplicates = {"a", "b", "c"}
        predictions = {"a", "b", "c", "d", "e"}
        assert classify_prediction(duplicates, predictions) == "TP"


class TestClustersToLabelsMinhash:
    def test_single_cluster(self) -> None:
        cluster_mapping = {0: 1, 1: 1, 2: 1}
        id_to_core_id = {0: "doc_a", 1: "doc_b", 2: "doc_c"}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b", "doc_c"}
        assert predictions["doc_b"] == {"doc_a", "doc_c"}
        assert predictions["doc_c"] == {"doc_a", "doc_b"}

    def test_multiple_clusters(self) -> None:
        cluster_mapping = {0: 1, 1: 1, 2: 2, 3: 2}
        id_to_core_id = {0: "doc_a", 1: "doc_b", 2: "doc_c", 3: "doc_d"}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b"}
        assert predictions["doc_b"] == {"doc_a"}
        assert predictions["doc_c"] == {"doc_d"}
        assert predictions["doc_d"] == {"doc_c"}

    def test_empty_cluster_mapping(self) -> None:
        cluster_mapping: dict = {}
        id_to_core_id: dict = {}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        assert predictions == {}

    def test_single_document_cluster(self) -> None:
        cluster_mapping = {0: 1}
        id_to_core_id = {0: "doc_a"}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == set()

    def test_missing_id_mapping(self) -> None:
        cluster_mapping = {0: 1, 1: 1, 2: 1}
        id_to_core_id = {0: "doc_a", 2: "doc_c"}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        assert "doc_a" in predictions
        assert "doc_c" in predictions
        assert 1 not in predictions

    def test_large_cluster(self) -> None:
        cluster_mapping = dict.fromkeys(range(10), 100)
        id_to_core_id = {i: f"doc_{i}" for i in range(10)}

        predictions = clusters_to_predictions_minhash(cluster_mapping, id_to_core_id)

        for i in range(10):
            doc_id = f"doc_{i}"
            expected = {f"doc_{j}" for j in range(10) if j != i}
            assert predictions[doc_id] == expected


class TestClustersToLabelsSimhash:
    def test_simple_parent_child(self) -> None:
        cluster_mapping = {1: 0}
        id_to_core_id = {0: "doc_a", 1: "doc_b"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b"}
        assert predictions["doc_b"] == {"doc_a"}

    def test_multiple_children_one_parent(self) -> None:
        cluster_mapping = {1: 0, 2: 0, 3: 0}
        id_to_core_id = {0: "doc_a", 1: "doc_b", 2: "doc_c", 3: "doc_d"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b", "doc_c", "doc_d"}
        assert predictions["doc_b"] == {"doc_a", "doc_c", "doc_d"}
        assert predictions["doc_c"] == {"doc_a", "doc_b", "doc_d"}
        assert predictions["doc_d"] == {"doc_a", "doc_b", "doc_c"}

    def test_multiple_separate_clusters(self) -> None:
        cluster_mapping = {1: 0, 3: 2}
        id_to_core_id = {0: "doc_a", 1: "doc_b", 2: "doc_c", 3: "doc_d"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b"}
        assert predictions["doc_b"] == {"doc_a"}
        assert predictions["doc_c"] == {"doc_d"}
        assert predictions["doc_d"] == {"doc_c"}

    def test_empty_cluster_mapping(self) -> None:
        cluster_mapping: dict = {}
        id_to_core_id: dict = {}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert predictions == {}

    def test_missing_id_mapping(self) -> None:
        cluster_mapping = {1: 0, 2: 0}
        id_to_core_id = {0: "doc_a"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert len(predictions) == 0

    def test_chain_of_parents(self) -> None:
        cluster_mapping = {1: 0, 2: 0, 3: 0}
        id_to_core_id = {0: "doc_parent", 1: "doc_child1", 2: "doc_child2", 3: "doc_child3"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        all_docs = {"doc_parent", "doc_child1", "doc_child2", "doc_child3"}
        for doc_id in all_docs:
            assert predictions[doc_id] == all_docs - {doc_id}

    def test_partial_id_mapping(self) -> None:
        cluster_mapping = {1: 0, 2: 0, 3: 0}
        id_to_core_id = {0: "doc_a", 1: "doc_b", 3: "doc_d"}

        predictions = clusters_to_predictions_simhash(cluster_mapping, id_to_core_id)

        assert predictions["doc_a"] == {"doc_b", "doc_d"}
        assert predictions["doc_b"] == {"doc_a", "doc_d"}
        assert predictions["doc_d"] == {"doc_a", "doc_b"}
