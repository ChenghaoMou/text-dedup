# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnusedCallResult=false
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

from text_dedup.config import Config
from text_dedup.config.algorithms.base import AlgorithmConfig
from text_dedup.config.io.input_configs import LocalInputConfig
from text_dedup.config.io.output_configs import OutputConfig
from text_dedup.data_sources.io import InvalidDatasetTypeError
from text_dedup.data_sources.io import load_dataset
from text_dedup.data_sources.io import save_dataset


class TestInvalidDatasetTypeError:
    @pytest.mark.parametrize(
        "data_type, expected_message",
        [
            (str, "Expecting Dataset object, loaded <class 'str'> instead"),
            (int, "Expecting Dataset object, loaded <class 'int'> instead"),
            (list, "Expecting Dataset object, loaded <class 'list'> instead"),
            (dict, "Expecting Dataset object, loaded <class 'dict'> instead"),
        ],
    )
    def test_invalid_dataset_type_error_with_different_types(self, data_type: type, expected_message: str) -> None:
        assert str(InvalidDatasetTypeError(data_type)) == expected_message


@pytest.fixture
def mock_config() -> Config:
    """Create a mock Config object for testing."""
    config = Mock(spec=Config)
    config.algorithm = Mock(spec=AlgorithmConfig)
    config.algorithm.internal_index_column = "__INDEX__"
    config.algorithm.cluster_column = "__CLUSTER__"
    config.algorithm.num_proc = 4
    config.input = Mock(spec=LocalInputConfig)
    config.input.read_arguments = {"path": "test_data", "split": "train"}
    config.output = Mock(spec=OutputConfig)
    config.output.output_dir = "test_output"
    config.output.save_clusters = False
    config.output.keep_index_column = False
    config.output.keep_cluster_column = False
    config.output.skip_filtering = False
    config.output.clean_cache = False
    return config


@pytest.fixture
def mock_dataset() -> Dataset:
    """Create a mock Dataset for testing."""
    mock_ds = Mock(spec=Dataset)
    mock_ds.map = Mock(return_value=mock_ds)
    mock_ds.remove_columns = Mock(return_value=mock_ds)
    mock_ds.save_to_disk = Mock()
    return mock_ds


class TestLoadDataset:
    @patch("text_dedup.data_sources.io.hf_load_dataset")
    def test_load_dataset_success(
        self,
        mock_hf_load: Mock,
        mock_config: Config,
        mock_dataset: Dataset,
    ) -> None:
        mock_hf_load.return_value = mock_dataset

        result = load_dataset(mock_config)

        # Verify dataset is loaded with correct arguments
        mock_hf_load.assert_called_once_with(**mock_config.input.read_arguments)

        # Verify dataset is mapped with indexing
        mock_dataset.map.assert_called_once()
        call_args = mock_dataset.map.call_args
        assert call_args[1]["with_indices"] is True
        assert call_args[1]["num_proc"] == mock_config.algorithm.num_proc
        assert call_args[1]["desc"] == "Indexing"

        # Verify lambda function adds internal index column
        lambda_func = call_args[0][0]
        result_dict = lambda_func({}, 42)
        assert result_dict == {"__INDEX__": 42}

        assert result == mock_dataset

    @patch("text_dedup.data_sources.io.hf_load_dataset")
    def test_load_dataset_invalid_type(self, mock_hf_load: Mock, mock_config: Config) -> None:
        # Return a non-Dataset object
        mock_hf_load.return_value = "not_a_dataset"

        with pytest.raises(InvalidDatasetTypeError, match="Expecting Dataset object, loaded <class 'str'> instead"):
            load_dataset(mock_config)

    @patch("text_dedup.data_sources.io.hf_load_dataset")
    def test_load_dataset_with_custom_internal_index_column(
        self,
        mock_hf_load: Mock,
        mock_config: Config,
        mock_dataset: Dataset,
    ) -> None:
        mock_config.algorithm.internal_index_column = "__CUSTOM_INDEX__"
        mock_hf_load.return_value = mock_dataset

        load_dataset(mock_config)

        # Verify lambda function uses custom internal index column
        call_args = mock_dataset.map.call_args
        lambda_func = call_args[0][0]
        result_dict = lambda_func({}, 99)
        assert result_dict == {"__CUSTOM_INDEX__": 99}


class TestSaveDataset:
    def test_save_dataset_basic(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10, 2: 20}

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify columns are removed correctly
            expected_columns_to_remove = {"__INDEX__", "__CLUSTER__"}
            mock_dataset.remove_columns.assert_called_once_with(list(expected_columns_to_remove))

            # Verify dataset is saved to disk
            mock_dataset.save_to_disk.assert_called_once_with(temp_dir, num_proc=4)

    def test_save_dataset_with_clusters_save(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10, 2: 20, 3: 30}
        mock_config.output.save_clusters = True
        mock_config.output.keep_index_column = True  # Explicitly set to True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify clusters are saved to pickle file
            clusters_file = Path(temp_dir) / "clusters.pickle"
            assert clusters_file.exists()

            with open(clusters_file, "rb") as f:
                saved_clusters = pickle.load(f)  # noqa: S301
            assert saved_clusters == clusters

            # When saving clusters and keeping index column, no columns are removed
            # because both index and cluster columns are kept
            mock_dataset.remove_columns.assert_not_called()

    @patch("text_dedup.data_sources.io.log")
    def test_save_dataset_clusters_without_index_column(
        self, mock_log: Mock, mock_config: Config, mock_dataset: Dataset
    ) -> None:
        clusters = {1: 10}
        mock_config.output.save_clusters = True
        mock_config.output.keep_index_column = False

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify warning is logged and keep_index_column is set to True
            mock_log.warning.assert_called_once_with("Saving clusters requires `--keep-index-column`, turning it on")
            assert mock_config.output.keep_index_column is True

            # Since keep_index_column was set to True and save_clusters is True,
            # both columns are kept, so no remove_columns call
            mock_dataset.remove_columns.assert_not_called()

    def test_save_dataset_clusters_save_forces_cluster_column_kept(
        self, mock_config: Config, mock_dataset: Dataset
    ) -> None:
        clusters = {1: 10, 2: 20}
        mock_config.output.save_clusters = True
        mock_config.output.keep_index_column = False  # Don't keep index column
        mock_config.output.keep_cluster_column = False  # Explicitly don't keep cluster column

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify clusters are saved to pickle file
            clusters_file = Path(temp_dir) / "clusters.pickle"
            assert clusters_file.exists()

            # Since save_clusters=True forces keep_index_column=True and cluster column is kept,
            # no columns should be removed
            mock_dataset.remove_columns.assert_not_called()
            assert mock_config.output.keep_index_column is True

    def test_save_dataset_keep_index_column(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}
        mock_config.output.keep_index_column = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify only cluster column is removed when keeping index column
            expected_columns_to_remove = {"__CLUSTER__"}
            mock_dataset.remove_columns.assert_called_once_with(list(expected_columns_to_remove))

    def test_save_dataset_keep_cluster_column(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}
        mock_config.output.keep_cluster_column = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify only index column is removed when keeping cluster column
            expected_columns_to_remove = {"__INDEX__"}
            mock_dataset.remove_columns.assert_called_once_with(list(expected_columns_to_remove))

    def test_save_dataset_keep_both_columns(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}
        mock_config.output.keep_index_column = True
        mock_config.output.keep_cluster_column = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify no columns are removed when keeping both
            mock_dataset.remove_columns.assert_not_called()

    def test_save_dataset_no_columns_to_remove(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}
        mock_config.output.keep_index_column = True
        mock_config.output.keep_cluster_column = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # When no columns to remove, remove_columns should not be called
            mock_dataset.remove_columns.assert_not_called()
            mock_dataset.save_to_disk.assert_called_once_with(temp_dir, num_proc=4)

    def test_save_dataset_with_custom_column_names(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}
        mock_config.algorithm.internal_index_column = "__CUSTOM_INDEX__"
        mock_config.algorithm.cluster_column = "__CUSTOM_CLUSTER__"

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify custom column names are used
            expected_columns_to_remove = {"__CUSTOM_INDEX__", "__CUSTOM_CLUSTER__"}
            mock_dataset.remove_columns.assert_called_once_with(list(expected_columns_to_remove))

    def test_save_dataset_with_kwargs(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10}

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            # Test that additional kwargs are accepted but unused
            save_dataset(
                mock_config, final_data=mock_dataset, clusters=clusters, extra_param="unused", another_param=42
            )

            # Should still work normally
            mock_dataset.save_to_disk.assert_called_once_with(temp_dir, num_proc=4)

    def test_save_dataset_pathlib_path(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters = {1: 10, 2: 20}
        mock_config.output.save_clusters = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify Path object is handled correctly
            clusters_file = Path(temp_dir) / "clusters.pickle"
            assert clusters_file.exists()

            # Verify the pickle file has correct protocol
            with open(clusters_file, "rb") as f:
                saved_clusters = pickle.load(f)  # noqa: S301
            assert saved_clusters == clusters

    def test_save_dataset_empty_clusters(self, mock_config: Config, mock_dataset: Dataset) -> None:
        clusters: dict[int, int] = {}
        mock_config.output.save_clusters = True

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.output.output_dir = temp_dir

            save_dataset(mock_config, final_data=mock_dataset, clusters=clusters)

            # Verify empty clusters dict is saved correctly
            clusters_file = Path(temp_dir) / "clusters.pickle"
            assert clusters_file.exists()

            with open(clusters_file, "rb") as f:
                saved_clusters = pickle.load(f)  # noqa: S301
            assert saved_clusters == {}
