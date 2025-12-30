import pytest

from benchmarks.benchmark_core import evaluate_predictions
from benchmarks.benchmark_core import prepare_ground_truth


class TestPrepareGroundTruth:
    def test_simple_dataset(self) -> None:
        dataset = [
            {"core_id": "1", "labelled_duplicates": ["2", "3"]},
            {"core_id": "2", "labelled_duplicates": ["1"]},
            {"core_id": "3", "labelled_duplicates": None},
        ]

        id_to_core_id, labels = prepare_ground_truth(dataset)

        assert id_to_core_id == {0: "1", 1: "2", 2: "3"}
        assert labels["1"] == {"2", "3"}
        assert labels["2"] == {"1"}
        assert labels["3"] == set()

    def test_empty_dataset(self) -> None:
        dataset: list = []

        id_to_core_id, labels = prepare_ground_truth(dataset)

        assert id_to_core_id == {}
        assert labels == {}

    def test_no_duplicates(self) -> None:
        dataset = [
            {"core_id": "1", "labelled_duplicates": None},
            {"core_id": "2", "labelled_duplicates": []},
            {"core_id": "3", "labelled_duplicates": None},
        ]

        id_to_core_id, labels = prepare_ground_truth(dataset)

        assert labels["1"] == set()
        assert labels["2"] == set()
        assert labels["3"] == set()

    def test_all_duplicates(self) -> None:
        dataset = [
            {"core_id": "1", "labelled_duplicates": ["2", "3", "4"]},
            {"core_id": "2", "labelled_duplicates": ["1", "3", "4"]},
            {"core_id": "3", "labelled_duplicates": ["1", "2", "4"]},
            {"core_id": "4", "labelled_duplicates": ["1", "2", "3"]},
        ]

        id_to_core_id, labels = prepare_ground_truth(dataset)

        assert len(labels) == 4
        assert labels["1"] == {"2", "3", "4"}
        assert labels["2"] == {"1", "3", "4"}

    def test_mixed_duplicates(self) -> None:
        dataset = [
            {"core_id": "1", "labelled_duplicates": ["2"]},
            {"core_id": "2", "labelled_duplicates": ["1"]},
            {"core_id": "3", "labelled_duplicates": None},
        ]

        id_to_core_id, labels = prepare_ground_truth(dataset)

        assert labels["1"] == {"2"}
        assert labels["2"] == {"1"}
        assert labels["3"] == set()


class TestEvaluatePredictions:
    def test_perfect_predictions(self) -> None:
        labels = {
            "1": {"2", "3"},
            "2": {"1", "3"},
            "3": {"1", "2"},
            "4": set(),
        }
        predictions = {
            "1": {"2", "3"},
            "2": {"1", "3"},
            "3": {"1", "2"},
            "4": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_duplicates"] == 1.0
        assert metrics["recall_duplicates"] == 1.0
        assert metrics["precision_non_duplicates"] == 1.0
        assert metrics["recall_non_duplicates"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_all_false_positives(self) -> None:
        labels = {
            "1": set(),
            "2": set(),
            "3": set(),
        }
        predictions = {
            "1": {"2"},
            "2": {"1"},
            "3": {"1"},
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision_duplicates"] == 0.0
        assert metrics["recall_duplicates"] == 0.0
        assert metrics["class_distribution"]["FP"] == 3

    def test_all_false_negatives(self) -> None:
        labels = {
            "1": {"2"},
            "2": {"1"},
            "3": {"1"},
        }
        predictions = {
            "1": set(),
            "2": set(),
            "3": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 0.0
        assert metrics["recall_duplicates"] == 0.0
        assert metrics["class_distribution"]["FN"] == 3

    def test_mixed_results(self) -> None:
        labels = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": set(),
        }
        predictions = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": {"1"},
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 0.75
        assert metrics["class_distribution"]["TP"] == 2
        assert metrics["class_distribution"]["TN"] == 1
        assert metrics["class_distribution"]["FP"] == 1

    def test_partial_recall(self) -> None:
        labels = {
            "1": {"2", "3", "4"},
            "2": set(),
        }
        predictions = {
            "1": {"2"},
            "2": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["recall_duplicates"] < 1.0
        assert metrics["class_distribution"]["FP"] == 1

    def test_empty_labels_and_predictions(self) -> None:
        labels = {"1": set()}
        predictions = {"1": set()}

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 1.0

    def test_precision_recall_metrics_exist(self) -> None:
        labels = {
            "1": {"2"},
            "2": {"1"},
        }
        predictions = {
            "1": {"2"},
            "2": {"1"},
        }

        metrics = evaluate_predictions(labels, predictions)

        assert "precision_duplicates" in metrics
        assert "recall_duplicates" in metrics
        assert "precision_non_duplicates" in metrics
        assert "recall_non_duplicates" in metrics
        assert metrics["precision_duplicates"] >= 0.0
        assert metrics["recall_duplicates"] >= 0.0

    def test_non_duplicate_metrics(self) -> None:
        labels = {
            "1": set(),
            "2": set(),
            "3": {"4"},
            "4": set(),
        }
        predictions = {
            "1": set(),
            "2": set(),
            "3": set(),
            "4": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["precision_non_duplicates"] > 0.5
        assert metrics["recall_non_duplicates"] > 0.5

    def test_macro_f1_calculation(self) -> None:
        labels = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": set(),
        }
        predictions = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        dup_prec = metrics["precision_duplicates"]
        non_dup_prec = metrics["precision_non_duplicates"]
        expected_macro = (dup_prec + non_dup_prec) / 2
        assert metrics["macro_f1"] == pytest.approx(expected_macro)

    def test_class_distribution_counts(self) -> None:
        labels = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": {"5"},
            "5": set(),
        }
        predictions = {
            "1": {"2"},
            "2": {"1"},
            "3": set(),
            "4": set(),
            "5": {"6"},
        }

        metrics = evaluate_predictions(labels, predictions)

        class_dist = metrics["class_distribution"]
        assert class_dist["TP"] == 2
        assert class_dist["TN"] == 1
        assert class_dist.get("FN", 0) + class_dist.get("FP", 0) == 2

    def test_single_document(self) -> None:
        labels = {"1": set()}
        predictions = {"1": set()}

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 1.0
        assert metrics["class_distribution"]["TN"] == 1

    def test_all_true_negatives(self) -> None:
        labels = {
            "1": set(),
            "2": set(),
            "3": set(),
        }
        predictions = {
            "1": set(),
            "2": set(),
            "3": set(),
        }

        metrics = evaluate_predictions(labels, predictions)

        assert metrics["accuracy"] == 1.0
        assert metrics["class_distribution"]["TN"] == 3
        assert metrics["precision_non_duplicates"] == 1.0
        assert metrics["recall_non_duplicates"] == 1.0
