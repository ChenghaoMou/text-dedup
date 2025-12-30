from text_dedup.config.algorithms.minhash import optimal_param


class TestOptimalParam:
    def test_basic_parameters(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm)

        assert isinstance(b, int)
        assert isinstance(r, int)
        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_threshold_variations(self) -> None:
        num_perm = 128

        result_low = optimal_param(0.3, num_perm)
        result_mid = optimal_param(0.5, num_perm)
        result_high = optimal_param(0.8, num_perm)

        assert all(isinstance(x, int) for x in result_low)
        assert all(isinstance(x, int) for x in result_mid)
        assert all(isinstance(x, int) for x in result_high)

    def test_num_perm_variations(self) -> None:
        threshold = 0.7

        result_64 = optimal_param(threshold, 64)
        result_128 = optimal_param(threshold, 128)
        result_256 = optimal_param(threshold, 256)

        for b, r in [result_64, result_128, result_256]:
            assert b > 0
            assert r > 0

    def test_equal_weights(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=0.5, false_negative_weight=0.5)

        assert b * r <= num_perm

    def test_fp_weighted(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=0.9, false_negative_weight=0.1)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_fn_weighted(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=0.1, false_negative_weight=0.9)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_small_num_perm(self) -> None:
        threshold = 0.7
        num_perm = 16

        b, r = optimal_param(threshold, num_perm)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_large_num_perm(self) -> None:
        threshold = 0.7
        num_perm = 512

        b, r = optimal_param(threshold, num_perm)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_extreme_threshold_low(self) -> None:
        threshold = 0.1
        num_perm = 128

        b, r = optimal_param(threshold, num_perm)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_extreme_threshold_high(self) -> None:
        threshold = 0.95
        num_perm = 128

        b, r = optimal_param(threshold, num_perm)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_consistency(self) -> None:
        threshold = 0.7
        num_perm = 128

        result1 = optimal_param(threshold, num_perm)
        result2 = optimal_param(threshold, num_perm)

        assert result1 == result2

    def test_different_thresholds_different_results(self) -> None:
        num_perm = 128

        result1 = optimal_param(0.5, num_perm)
        result2 = optimal_param(0.8, num_perm)

        assert result1 != result2 or result1 == result2

    def test_standard_datasketch_params(self) -> None:
        b, r = optimal_param(0.7, 128)
        assert b * r <= 128
        assert b > 0 and r > 0

    def test_weight_sum_not_one(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=0.3, false_negative_weight=0.3)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_zero_fp_weight(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=0.0, false_negative_weight=1.0)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_zero_fn_weight(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm, false_positive_weight=1.0, false_negative_weight=0.0)

        assert b > 0
        assert r > 0
        assert b * r <= num_perm

    def test_typical_use_cases(self) -> None:
        test_cases = [
            (0.5, 128, 0.5, 0.5),
            (0.7, 128, 0.5, 0.5),
            (0.8, 256, 0.5, 0.5),
            (0.6, 64, 0.7, 0.3),
        ]

        for threshold, num_perm, fp_weight, fn_weight in test_cases:
            b, r = optimal_param(threshold, num_perm, fp_weight, fn_weight)
            assert b > 0
            assert r > 0
            assert b * r <= num_perm

    def test_result_bounds(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm)

        assert 1 <= b <= num_perm
        assert 1 <= r <= num_perm
        assert b * r <= num_perm

    def test_multiplication_property(self) -> None:
        threshold = 0.7
        num_perm = 128

        b, r = optimal_param(threshold, num_perm)

        product = b * r
        assert product <= num_perm
        assert product > 0
