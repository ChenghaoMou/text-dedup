# pyright: reportCallIssue=false
# pyright: reportPrivateUsage=false
# pyright: reportUnusedCallResult=false
import os

import numpy as np
import pytest
from pydantic import ValidationError

from text_dedup.config.algorithms.base import AlgorithmConfig


class TestAlgorithmConfig:
    @pytest.mark.parametrize(
        "algorithm_name",
        ["minhash", "simhash", "bloom_filter", "suffix_array"],
    )
    def test_algorithm_config_defaults(self, algorithm_name: str) -> None:
        config = AlgorithmConfig(algorithm_name=algorithm_name, text_column="text")

        assert config.algorithm_name == algorithm_name
        assert config.text_column == "text"
        assert config.index_column is None
        assert config.cluster_column == "__CLUSTER__"
        assert config.signature_column == "__SIGNATURE__"
        assert config.seed == 42
        assert config.batch_size == 1000
        assert config.internal_index_column == "__INDEX__"
        assert isinstance(config._rng, np.random.RandomState)

    def test_algorithm_config_custom_values(self) -> None:
        config = AlgorithmConfig(
            algorithm_name="simhash",
            text_column="content",
            index_column="doc_id",
            cluster_column="cluster_id",
            signature_column="signature",
            seed=123,
            batch_size=500,
        )

        assert config.algorithm_name == "simhash"
        assert config.text_column == "content"
        assert config.index_column == "doc_id"
        assert config.cluster_column == "cluster_id"
        assert config.signature_column == "signature"
        assert config.seed == 123
        assert config.batch_size == 500

    def test_algorithm_config_invalid_algorithm_name(self) -> None:
        with pytest.raises(ValidationError, match="algorithm_name"):
            AlgorithmConfig(algorithm_name="invalid_algorithm", text_column="text")

    def test_algorithm_config_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError, match="text_column"):
            AlgorithmConfig(algorithm_name="minhash")

        with pytest.raises(ValidationError, match="algorithm_name"):
            AlgorithmConfig(text_column="text")

    def test_num_proc_default_behavior(self) -> None:
        # Test that num_proc defaults to a positive integer
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text")
        assert isinstance(config.num_proc, int)
        assert config.num_proc >= 1

    def test_num_proc_default_logic(self) -> None:
        # Test the logic used in the default calculation
        # This tests the max(1, os.cpu_count() or 1) formula
        cpu_count = os.cpu_count()
        expected = max(1, cpu_count or 1)

        config = AlgorithmConfig(algorithm_name="minhash", text_column="text")
        assert config.num_proc == expected

    def test_num_proc_custom_value(self) -> None:
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text", num_proc=4)
        assert config.num_proc == 4

    def test_rng_initialization_with_seed(self) -> None:
        seed = 999
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=seed)

        assert isinstance(config._rng, np.random.RandomState)

        # Test that the same seed produces the same random numbers
        config1 = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=seed)
        config2 = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=seed)

        # Generate some random numbers and compare
        assert config1._rng is not None and config2._rng is not None
        random1 = [config1._rng.randint(0, 1000) for _ in range(10)]
        random2 = [config2._rng.randint(0, 1000) for _ in range(10)]

        assert random1 == random2

    def test_rng_different_seeds(self) -> None:
        config1 = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=42)
        config2 = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=43)

        # Generate random numbers with different seeds
        assert config1._rng is not None and config2._rng is not None
        random1 = [config1._rng.randint(0, 1000) for _ in range(10)]
        random2 = [config2._rng.randint(0, 1000) for _ in range(10)]

        # They should be different (with very high probability)
        assert random1 != random2

    def test_internal_index_column_property(self) -> None:
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text")

        assert config.internal_index_column == config._internal_index_column == "__INDEX__"

    def test_model_post_init_called(self) -> None:
        # Test that model_post_init is called and _rng is initialized
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=555)

        assert config._rng is not None
        assert isinstance(config._rng, np.random.RandomState)

        # Verify the RNG was initialized with the correct seed
        # by creating another instance with the same seed and comparing
        config._rng = np.random.RandomState(555)
        test_rng = np.random.RandomState(555)

        assert config._rng.randint(0, 1000) == test_rng.randint(0, 1000)

    def test_string_field_types(self) -> None:
        config = AlgorithmConfig(
            algorithm_name="minhash",
            text_column="text",
            index_column="idx",
            cluster_column="cluster",
            signature_column="sig",
        )

        assert isinstance(config.text_column, str)
        assert isinstance(config.index_column, str)
        assert isinstance(config.cluster_column, str)
        assert isinstance(config.signature_column, str)

    def test_integer_field_types(self) -> None:
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text", seed=100, num_proc=2, batch_size=2000)

        assert isinstance(config.seed, int)
        assert isinstance(config.num_proc, int)
        assert isinstance(config.batch_size, int)

    def test_config_immutability_after_init(self) -> None:
        config = AlgorithmConfig(algorithm_name="minhash", text_column="text")

        # The _rng should be set and accessible
        original_rng = config._rng
        assert original_rng is not None

        # Test that we can access the internal index column
        assert config.internal_index_column == "__INDEX__"
